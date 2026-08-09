"""Construction of distance data from vectors: condensed vectors and full matrices.

Every build comes in a sequential and a parallel variant with bit-identical output — the parallel
fills run the same pair arithmetic under the same fastmath flags and write each element exactly
once, so thread count cannot affect results.  `MAXDIV_PARALLEL_BUILD` picks the variant (parallel
unless set to ``"0"``).

The parallel fills cut the columns into fixed-width blocks and parallelize the row loop within
each block: every row above a block computes exactly one block-width of pairs, so prange can
split the work evenly across threads — parallelizing the outer row loop over the whole i<j pair
triangle instead leaves one thread with roughly twice the work of the average.  Condensed fills
address each row's output segment by the closed-form condensed offset, since a parallel loop
cannot share a running index.

Cosine is the one metric with dedicated condensed fills: its rows are normalized once in the
entry point, and those fills deliberately carry no fastmath flags, so condensed cosine values are
bit-reproducible across machines and numba versions, while the matrix fills compute every metric
(cosine included) under `reassoc`/`contract`.

Every fill writes into a buffer the caller supplies, and each entry point allocates one only when
none is given.  Fills write the off-diagonal pairs only, so the entry points zero the diagonal
themselves — a supplied buffer carries no guarantee of starting at zero.
"""

import os

import numba
import numpy as np
from numpy.typing import NDArray

from ._compute import _METRIC_KINDS, _l2sq_pair, _metric_pair, normalize_rows, validate_cosine_vectors
from ._enum import DistanceMetric

# Width in columns of the blocks the parallel fills cut the pair space into.
BUILD_BLOCK_WIDTH = 64

# Condensed distances arrive from a DistanceStore, whose arrays are read-only views; DISTANCE_STORE_TYPE
# covers why a read-only parameter accepts writable arguments too.  Signature strings cannot spell a
# read-only array type, so the fill taking one is given a signature object.
_READONLY_F32_1D = numba.types.Array(numba.float32, 1, "C", readonly=True)


def parallel_build_enabled() -> bool:
    """Return whether distance builds may use multiple threads (default: enabled)."""
    return os.environ.get("MAXDIV_PARALLEL_BUILD", "1") != "0"


# =================================================================================================
#  Public entry points
# =================================================================================================
def compute_pdist(
    vectors: NDArray[np.float32], metric: DistanceMetric, out: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Compute the pair-wise distances between a set of n vectors in d dimensions.

    Computed directly in float32 — each pair accumulates in float64 across the d dimensions and
    narrows to float32 on store — avoiding the float64 distance matrix scipy would otherwise
    materialize and cast (a ~3x transient in peak setup memory).

    :param vectors: (n x d ndarray) A set of n vectors in d dimensions.
    :param metric: (DistanceMetric) The distance metric to use.
    :param out: ((n*(n-1))//2 ndarray) buffer to fill, allocated here when not given.
    :return: ((n*(n-1))//2 ndarray) condensed pair-wise distance vector, in scipy's layout: the
                                         (i,j)-distance for i<j sits at the offset given by `_condensed_index`.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    if out is None:
        out = np.empty((n * (n - 1)) // 2, dtype=np.float32)
    if metric == DistanceMetric.COSINE:
        validate_cosine_vectors(vectors)
        normalized = normalize_rows(vectors)
        if parallel_build_enabled():
            _fill_pdist_cos_parallel(normalized, np.int64(BUILD_BLOCK_WIDTH), out)
        else:
            _fill_pdist_cos(normalized, out)
        return out
    if parallel_build_enabled():
        _fill_pdist_parallel(vectors, _METRIC_KINDS[metric], np.int64(BUILD_BLOCK_WIDTH), out)
    else:
        _fill_pdist(vectors, _METRIC_KINDS[metric], out)
    return out


def compute_full_matrix(
    vectors: NDArray[np.float32], metric: DistanceMetric, out: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Compute the full (n, n) pair-wise distance matrix, exactly symmetric by construction.

    Each pair is computed once through the same pair arithmetic the condensed and lazy paths use,
    and written to both halves.

    :param vectors: (n x d ndarray) A set of n vectors in d dimensions.
    :param metric: (DistanceMetric) The distance metric to use.
    :param out: ((n, n) ndarray) buffer to fill, allocated here when not given.
    :return: ((n, n) ndarray) full pairwise-distance matrix, float32 C-contiguous.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if metric == DistanceMetric.COSINE:
        validate_cosine_vectors(vectors)
        vectors = normalize_rows(vectors)
    out = _matrix_buffer(out, vectors.shape[0])
    if parallel_build_enabled():
        _fill_matrix_parallel(vectors, _METRIC_KINDS[metric], np.int64(BUILD_BLOCK_WIDTH), out)
    else:
        _fill_matrix(vectors, _METRIC_KINDS[metric], out)
    return out


def expand_condensed(
    condensed: NDArray[np.float32], n: int, out: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Expand a condensed distance vector into a full (n, n) matrix, each value written to both halves.

    :param condensed: ((n*(n-1))//2 ndarray) condensed pairwise distances, float32 C-contiguous.
    :param n: (int) number of items.
    :param out: ((n, n) ndarray) buffer to fill, allocated here when not given.
    :return: ((n, n) ndarray) full pairwise-distance matrix, bit-equal to the condensed source.
    """
    out = _matrix_buffer(out, n)
    _fill_matrix_from_condensed(condensed, np.int32(n), out)
    return out


def _matrix_buffer(out: NDArray[np.float32] | None, n: int) -> NDArray[np.float32]:
    """Return the (n, n) buffer the matrix fills write into, allocated when not supplied."""
    if out is None:
        out = np.empty((n, n), dtype=np.float32)
    np.fill_diagonal(out, np.float32(0.0))
    return out


# =================================================================================================
#  Condensed fills
# =================================================================================================
@numba.njit("void(float32[:, ::1], int32, float32[::1])", cache=True, fastmath={"reassoc", "contract"})
def _fill_pdist(vectors: NDArray[np.float32], metric_kind: np.int32, out: NDArray[np.float32]) -> None:
    """Write condensed distances for every metric except cosine into `out`, sequentially."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = _metric_pair(vectors, metric_kind, np.int32(i), np.int32(j))
            idx += 1


@numba.njit(
    "void(float32[:, ::1], int32, int64, float32[::1])", parallel=True, cache=True, fastmath={"reassoc", "contract"}
)
def _fill_pdist_parallel(
    vectors: NDArray[np.float32], metric_kind: np.int32, block_width: np.int64, out: NDArray[np.float32]
) -> None:
    """Write condensed distances for every metric except cosine into `out`, in parallel."""
    n = vectors.shape[0]
    for j_block in range(0, n, block_width):
        j_end = min(j_block + block_width, n)
        for i in numba.prange(j_end):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
            base = np.int64(i) * n - (np.int64(i) * (i + 1)) // 2 - i - 1
            for j in range(max(j_block, i + 1), j_end):
                out[base + j] = _metric_pair(vectors, metric_kind, np.int32(i), np.int32(j))


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True)
def _fill_pdist_cos(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write condensed cosine distances of pre-normalized rows into `out`, sequentially.

    Computed as ``0.5 * ||x^ - y^||^2`` on unit-normalized vectors, which equals ``1 - cos(x, y)``
    algebraically but — unlike the dot-product form — is non-negative by construction and exactly
    0.0 for identical vectors.
    """
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(0.5 * _l2sq_pair(vectors, i, j))
            idx += 1


@numba.njit("void(float32[:, ::1], int64, float32[::1])", parallel=True, cache=True)
def _fill_pdist_cos_parallel(vectors: NDArray[np.float32], block_width: np.int64, out: NDArray[np.float32]) -> None:
    """Write condensed cosine distances of pre-normalized rows into `out`, in parallel."""
    n = vectors.shape[0]
    for j_block in range(0, n, block_width):
        j_end = min(j_block + block_width, n)
        for i in numba.prange(j_end):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
            base = np.int64(i) * n - (np.int64(i) * (i + 1)) // 2 - i - 1
            for j in range(max(j_block, i + 1), j_end):
                out[base + j] = np.float32(0.5 * _l2sq_pair(vectors, np.int32(i), np.int32(j)))


# =================================================================================================
#  Full-matrix fills
# =================================================================================================
@numba.njit("void(float32[:, ::1], int32, float32[:, ::1])", cache=True, fastmath={"reassoc", "contract"})
def _fill_matrix(vectors: NDArray[np.float32], metric_kind: np.int32, out: NDArray[np.float32]) -> None:
    """Fill a full (n, n) distance matrix from vectors, sequentially; each pair written to both halves."""
    n = vectors.shape[0]
    for i in np.arange(n, dtype=np.int32):
        for j in np.arange(i + 1, n, dtype=np.int32):
            value = _metric_pair(vectors, metric_kind, i, j)
            out[i, j] = value
            out[j, i] = value


@numba.njit(
    "void(float32[:, ::1], int32, int64, float32[:, ::1])",
    parallel=True,
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _fill_matrix_parallel(
    vectors: NDArray[np.float32], metric_kind: np.int32, block_width: np.int64, out: NDArray[np.float32]
) -> None:
    """Fill a full (n, n) distance matrix from vectors, in parallel; each pair written to both halves."""
    n = vectors.shape[0]
    for j_block in range(0, n, block_width):
        j_end = min(j_block + block_width, n)
        for i in numba.prange(j_end):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
            for j in range(max(j_block, np.int64(i) + 1), j_end):
                value = _metric_pair(vectors, metric_kind, np.int32(i), np.int32(j))
                out[i, j] = value
                out[j, i] = value


@numba.njit(numba.void(_READONLY_F32_1D, numba.int32, numba.float32[:, ::1]), cache=True)
def _fill_matrix_from_condensed(condensed: NDArray[np.float32], n: np.int32, out: NDArray[np.float32]) -> None:
    """Expand a condensed distance vector into a full (n, n) matrix: each value written to both halves."""
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            value = condensed[idx]
            idx += 1
            out[i, j] = value
            out[j, i] = value
