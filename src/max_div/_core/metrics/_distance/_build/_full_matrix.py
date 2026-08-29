"""The full-matrix build writes an (n, n) matrix, either from vectors or by expanding a condensed vector.

Each pair is computed once and written to both halves, so the matrix is exactly symmetric by
construction rather than by a repair pass — kernels read whichever of (i, j)/(j, i) suits their
access pattern, so the two halves must be bit-equal.  Every fill also zeroes the diagonal it never
computes, so a fill leaves a complete matrix whatever the buffer held beforehand.

Unlike the condensed cosine fills, these compute every metric — cosine included — under
`reassoc`/`contract`.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics._distance._metric import (
    DistanceMetric,
    _metric_pair,
    normalize_rows,
    validate_cosine_vectors,
)

from ._common import (
    BUILD_BLOCK_WIDTH,
    READONLY_F32_1D,
    READONLY_F32_2D,
    WRITABLE_F32_2D,
    parallel_build_enabled,
)


# =================================================================================================
#  Entry points
# =================================================================================================
def compute_full_matrix(
    vectors: NDArray[np.float32], metric: DistanceMetric, out: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Compute the full (n, n) pair-wise distance matrix, exactly symmetric by construction.

    Each pair is computed once through the same pair arithmetic the condensed and lazy paths use,
    and written to both halves.

    Args:
        vectors: (n x d ndarray) A set of n vectors in d dimensions.
        metric: (DistanceMetric) The distance metric to use.
        out: ((n, n) ndarray) buffer to fill, allocated here when not given.

    Returns:
        ((n, n) ndarray) full pairwise-distance matrix, float32 C-contiguous — `out` itself
        whenever one was given, following numpy's convention for such a parameter.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if metric == DistanceMetric.cosine():
        validate_cosine_vectors(vectors)
        vectors = normalize_rows(vectors)
    out = _allocate_if_needed(out, vectors.shape[0])
    if parallel_build_enabled():
        _fill_matrix_parallel(vectors, np.int32(metric.kind), np.int64(BUILD_BLOCK_WIDTH), out)
    else:
        _fill_matrix(vectors, np.int32(metric.kind), out)
    return out


def expand_condensed(
    condensed: NDArray[np.float32], n: int, out: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Expand a condensed distance vector into a full (n, n) matrix, each value written to both halves.

    Args:
        condensed: ((n*(n-1))//2 ndarray) condensed pairwise distances, float32 C-contiguous.
        n: (int) number of items.
        out: ((n, n) ndarray) buffer to fill, allocated here when not given.

    Returns:
        ((n, n) ndarray) full pairwise-distance matrix, bit-equal to the condensed source —
        `out` itself whenever one was given, following numpy's convention for such a parameter.
    """
    out = _allocate_if_needed(out, n)
    _fill_matrix_from_condensed(condensed, np.int32(n), out)
    return out


def _allocate_if_needed(out: NDArray[np.float32] | None, n: int) -> NDArray[np.float32]:
    """Return `out`, or a fresh uninitialized (n, n) float32 buffer when none was given."""
    return np.empty((n, n), dtype=np.float32) if out is None else out


# =================================================================================================
#  Fills
# =================================================================================================
@numba.njit(numba.void(READONLY_F32_2D, numba.int32, WRITABLE_F32_2D), cache=True, fastmath={"reassoc", "contract"})
def _fill_matrix(vectors: NDArray[np.float32], metric_kind: np.int32, out: NDArray[np.float32]) -> None:
    """Fill a full (n, n) distance matrix from vectors, sequentially; each pair written to both halves."""
    n = vectors.shape[0]
    for i in np.arange(n, dtype=np.int32):
        out[i, i] = np.float32(0.0)
        for j in np.arange(i + 1, n, dtype=np.int32):
            value = _metric_pair(vectors, metric_kind, i, j)
            out[i, j] = value
            out[j, i] = value


@numba.njit(
    numba.void(READONLY_F32_2D, numba.int32, numba.int64, WRITABLE_F32_2D),
    parallel=True,
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _fill_matrix_parallel(
    vectors: NDArray[np.float32], metric_kind: np.int32, block_width: np.int64, out: NDArray[np.float32]
) -> None:
    """Fill a full (n, n) distance matrix from vectors, in parallel; each pair written to both halves."""
    n = vectors.shape[0]
    for i in numba.prange(n):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
        out[i, i] = np.float32(0.0)
    for j_block in range(0, n, block_width):
        j_end = min(j_block + block_width, n)
        for i in numba.prange(j_end):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
            for j in range(max(j_block, np.int64(i) + 1), j_end):
                value = _metric_pair(vectors, metric_kind, np.int32(i), np.int32(j))
                out[i, j] = value
                out[j, i] = value


@numba.njit(numba.void(READONLY_F32_1D, numba.int32, WRITABLE_F32_2D), cache=True)
def _fill_matrix_from_condensed(condensed: NDArray[np.float32], n: np.int32, out: NDArray[np.float32]) -> None:
    """Expand a condensed distance vector into a full (n, n) matrix: each value written to both halves."""
    idx = np.int64(0)
    for i in range(n):
        out[i, i] = np.float32(0.0)
        for j in range(i + 1, n):
            value = condensed[idx]
            idx += 1
            out[i, j] = value
            out[j, i] = value
