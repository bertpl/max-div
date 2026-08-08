"""Methods for computing pair-wise distances between vectors, along the lines of scipy's pdist."""

import numba
import numpy as np
from numpy.typing import NDArray

from ._enum import DistanceMetric
from ._parallel_build import BUILD_TILE, parallel_build_enabled

# Pair-kernel selector values, so njit kernels can branch on the metric without object-mode.
# Cosine holds pre-normalized vectors, so its pair read is the half-squared-L2 form on those.
_METRIC_KIND_L1 = np.int32(0)
_METRIC_KIND_L2 = np.int32(1)
_METRIC_KIND_L2S = np.int32(2)
_METRIC_KIND_COS = np.int32(3)
_METRIC_KIND_LINF = np.int32(4)

_METRIC_KINDS = {
    DistanceMetric.L1_MANHATTAN: _METRIC_KIND_L1,
    DistanceMetric.L2_EUCLIDEAN: _METRIC_KIND_L2,
    DistanceMetric.L2S_EUCLIDEAN_SQUARED: _METRIC_KIND_L2S,
    DistanceMetric.COSINE: _METRIC_KIND_COS,
    DistanceMetric.LINF_CHEBYSHEV: _METRIC_KIND_LINF,
}


# =================================================================================================
#  pdist computation
# =================================================================================================
def compute_pdist(vectors: NDArray[np.float32], metric: DistanceMetric) -> NDArray[np.float32]:
    """Compute the pair-wise distances between a set of n vectors in d dimensions.

    Computed directly in float32 by a numba kernel — each pair accumulates in float64 across the d
    dimensions and narrows to float32 on store — avoiding the float64 distance matrix scipy would
    otherwise materialize and cast (a ~3x transient in peak setup memory).

    :param vectors: (n x d ndarray) A set of n vectors in d dimensions.
    :param metric: (DistanceMetric) The distance metric to use.
    :return: ((n*(n-1))//2 ndarray) condensed pair-wise distance vector, in scipy's layout: the
                                         (i,j)-distance for i<j sits at the offset given by `_condensed_index`.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    out = np.empty((n * (n - 1)) // 2, dtype=np.float32)
    if parallel_build_enabled():
        if metric == DistanceMetric.COSINE:
            validate_cosine_vectors(vectors)
            _pdist_parallel_cos(normalize_rows(vectors), np.int64(BUILD_TILE), out)
        else:
            _pdist_parallel(vectors, _METRIC_KINDS[metric], np.int64(BUILD_TILE), out)
        return out
    match metric:
        case DistanceMetric.L1_MANHATTAN:
            _pdist_l1(vectors, out)
        case DistanceMetric.L2_EUCLIDEAN:
            _pdist_l2(vectors, out)
        case DistanceMetric.L2S_EUCLIDEAN_SQUARED:
            _pdist_l2s(vectors, out)
        case DistanceMetric.LINF_CHEBYSHEV:
            _pdist_linf(vectors, out)
        case DistanceMetric.COSINE:
            validate_cosine_vectors(vectors)
            _pdist_cos(vectors, out)
    return out


def validate_cosine_vectors(vectors: NDArray[np.float32]) -> None:
    """Raise ValueError if any vector is all-zero — cosine distance is undefined for zero vectors.

    :param vectors: (n x d ndarray) A set of n vectors in d dimensions.
    """
    zero_rows = np.flatnonzero(~vectors.any(axis=1))
    if zero_rows.size > 0:
        raise ValueError(
            f"Cosine distance is undefined for zero vectors; found an all-zero vector at row {zero_rows[0]}."
        )


# =================================================================================================
#  pdist kernels
# =================================================================================================
# These kernels sum one term per dimension.  Adding those terms in a different order gives a
# very slightly different answer in floating point, so by default the compiler must add them
# strictly left to right — one at a time, each waiting for the previous.  `reassoc` lifts that
# restriction ("reassociate" = regroup the additions), letting the compiler add several terms in
# parallel; `contract` lets a multiply and an add become one instruction.  Together they are what
# make these loops vectorize.
#
# Granted here rather than the full fastmath set, which would also assert that no value is ever
# infinite — untrue, since the separation arrays use +inf to mean "no selected neighbor yet".
#
# Accepted consequence: a distance is no longer a fixed function of this source, so values can
# differ in their last bits between machines or numba versions.  See the reproducibility section
# of the solver documentation.


@numba.njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _l1_pair(vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger) -> np.float64:
    """Return the L1 (Manhattan) distance between vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c]))
    return acc


@numba.njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _l2sq_pair(vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger) -> np.float64:
    """Return the squared L2 (Euclidean) distance between vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        diff = np.float64(vectors[i, c]) - np.float64(vectors[j, c])
        acc += diff * diff
    return acc


@numba.njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _linf_pair(vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger) -> np.float64:
    """Return the Linf (Chebyshev) distance between vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        diff = abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c]))
        if diff > acc:
            acc = diff
    return acc


@numba.njit(numba.float32(numba.float32[:, ::1], numba.int32, numba.int32, numba.int32), inline="always", cache=True)
def _lazy_pair(vectors: NDArray[np.float32], metric_kind: np.int32, i: np.int32, j: np.int32) -> np.float32:
    """Compute the distance between vectors i and j on demand, per the given pair-kernel selector.

    Each branch narrows exactly as the corresponding precomputing kernel does, so on-demand values
    are bit-equal to stored ones.
    """
    if metric_kind == _METRIC_KIND_L1:
        return np.float32(_l1_pair(vectors, i, j))
    if metric_kind == _METRIC_KIND_L2:
        return np.float32(np.sqrt(_l2sq_pair(vectors, i, j)))
    if metric_kind == _METRIC_KIND_L2S:
        return np.float32(_l2sq_pair(vectors, i, j))
    if metric_kind == _METRIC_KIND_LINF:
        return np.float32(_linf_pair(vectors, i, j))
    return np.float32(0.5 * _l2sq_pair(vectors, i, j))  # cosine: rows are pre-normalized


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True, fastmath={"reassoc", "contract"})
def _pdist_l1(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed L1 distances of `vectors` into pre-allocated `out`, in condensed i<j order."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(_l1_pair(vectors, i, j))
            idx += 1


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True, fastmath={"reassoc", "contract"})
def _pdist_l2(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed L2 distances of `vectors` into pre-allocated `out`, in condensed i<j order."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(np.sqrt(_l2sq_pair(vectors, i, j)))
            idx += 1


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True, fastmath={"reassoc", "contract"})
def _pdist_l2s(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed squared-L2 distances of `vectors` into pre-allocated `out`, in condensed i<j order."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(_l2sq_pair(vectors, i, j))
            idx += 1


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True, fastmath={"reassoc", "contract"})
def _pdist_linf(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed Linf distances of `vectors` into pre-allocated `out`, in condensed i<j order."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(_linf_pair(vectors, i, j))
            idx += 1


@numba.njit("float32[:, ::1](float32[:, ::1])", cache=True)
def normalize_rows(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a fresh float32 array with each row of `vectors` scaled to unit L2 norm.

    Norms accumulate in float64 and each element narrows to float32 on store, so the result is the
    exact normalization the cosine pairwise kernels operate on.  Rows must not be all-zero
    (`validate_cosine_vectors` guards the public entry points).
    """
    n = vectors.shape[0]
    d = vectors.shape[1]
    normalized = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        acc = np.float64(0.0)
        for c in range(d):
            acc += np.float64(vectors[i, c]) * np.float64(vectors[i, c])
        norm = np.sqrt(acc)
        for c in range(d):
            normalized[i, c] = np.float32(np.float64(vectors[i, c]) / norm)
    return normalized


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True)
def _pdist_cos(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed cosine distances of `vectors` into pre-allocated `out`, in condensed i<j order.

    Computed as ``0.5 * ||x^ - y^||^2`` on unit-normalized vectors, which equals ``1 - cos(x, y)``
    algebraically but — unlike the dot-product form — is non-negative by construction and exactly
    0.0 for identical vectors. Rows are normalized once (`normalize_rows`), so the pair loop
    reuses the squared-L2 accumulation.

    Vectors must contain no all-zero rows (`validate_cosine_vectors` guards the public entry point).
    """
    n = vectors.shape[0]
    normalized = normalize_rows(vectors)
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(0.5 * _l2sq_pair(normalized, i, j))
            idx += 1


# =================================================================================================
#  Parallel pdist kernels
# =================================================================================================
# Same pair arithmetic as the sequential kernels above — each mirrors its sequential counterpart's
# fastmath flags exactly, so parallel and sequential builds are bit-identical (each element is
# written exactly once, so thread count cannot affect the result either).  The column-block loop
# converts the triangular pair space into uniform slabs prange can split evenly; see
# `_parallel_build` for the tile rationale.  Row segments are addressed by the closed-form
# condensed offset, since a parallel loop cannot share the sequential kernels' running index.


@numba.njit(
    "void(float32[:, ::1], int32, int64, float32[::1])", parallel=True, cache=True, fastmath={"reassoc", "contract"}
)
def _pdist_parallel(
    vectors: NDArray[np.float32], metric_kind: np.int32, tile: np.int64, out: NDArray[np.float32]
) -> None:
    """Write condensed distances for the L1/L2/L2S/Linf metrics into `out`, in parallel."""
    n = vectors.shape[0]
    for j_block in range(0, n, tile):
        j_end = min(j_block + tile, n)
        for i in numba.prange(j_end):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
            base = np.int64(i) * n - (np.int64(i) * (i + 1)) // 2 - i - 1
            for j in range(max(j_block, i + 1), j_end):
                out[base + j] = _lazy_pair(vectors, metric_kind, np.int32(i), np.int32(j))


@numba.njit("void(float32[:, ::1], int64, float32[::1])", parallel=True, cache=True)
def _pdist_parallel_cos(vectors: NDArray[np.float32], tile: np.int64, out: NDArray[np.float32]) -> None:
    """Write condensed cosine distances into `out`, in parallel.

    Deliberately carries no fastmath flags, mirroring the sequential `_pdist_cos`.  Takes
    already-normalized rows (the caller normalizes once, as the sequential kernel does).
    """
    n = vectors.shape[0]
    for j_block in range(0, n, tile):
        j_end = min(j_block + tile, n)
        for i in numba.prange(j_end):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
            base = np.int64(i) * n - (np.int64(i) * (i + 1)) // 2 - i - 1
            for j in range(max(j_block, i + 1), j_end):
                out[base + j] = np.float32(0.5 * _l2sq_pair(vectors, np.int32(i), np.int32(j)))
