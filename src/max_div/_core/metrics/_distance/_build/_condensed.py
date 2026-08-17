"""The condensed build writes every pair's distance into a single vector, in scipy's layout.

The parallel fills address each row's output segment by the closed-form condensed offset, since a
parallel loop cannot share a running index.

Cosine is the one metric with dedicated fills here: its rows are normalized once in the entry point,
and those fills deliberately carry no fastmath flags, so condensed cosine values are bit-reproducible
across machines and numba versions.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics._distance._metric import (
    _METRIC_KINDS,
    DistanceMetric,
    _l2sq_pair,
    _metric_pair,
    normalize_rows,
    validate_cosine_vectors,
)

from ._common import BUILD_BLOCK_WIDTH, READONLY_F32_2D, WRITABLE_F32_1D, parallel_build_enabled


# =================================================================================================
#  Entry point
# =================================================================================================
def compute_pdist(
    vectors: NDArray[np.float32], metric: DistanceMetric, out: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Compute the pair-wise distances between a set of n vectors in d dimensions.

    Computed directly in float32 — each pair accumulates in float64 across the d dimensions and
    narrows to float32 on store — avoiding the float64 distance matrix scipy would otherwise
    materialize and cast (a ~3x transient in peak setup memory).

    Args:
        vectors: (n x d ndarray) A set of n vectors in d dimensions.
        metric: (DistanceMetric) The distance metric to use.
        out: ((n*(n-1))//2 ndarray) buffer to fill, allocated here when not given.

    Returns:
        ((n*(n-1))//2 ndarray) condensed pair-wise distance vector, in scipy's layout: the
        (i,j)-distance for i<j sits at the offset given by `_condensed_index`.  This is `out`
        itself whenever one was given, following numpy's convention for such a parameter.
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


# =================================================================================================
#  Fills
# =================================================================================================
@numba.njit(numba.void(READONLY_F32_2D, numba.int32, WRITABLE_F32_1D), cache=True, fastmath={"reassoc", "contract"})
def _fill_pdist(vectors: NDArray[np.float32], metric_kind: np.int32, out: NDArray[np.float32]) -> None:
    """Write condensed distances for every metric except cosine into `out`, sequentially."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = _metric_pair(vectors, metric_kind, np.int32(i), np.int32(j))
            idx += 1


@numba.njit(
    numba.void(READONLY_F32_2D, numba.int32, numba.int64, WRITABLE_F32_1D),
    parallel=True,
    cache=True,
    fastmath={"reassoc", "contract"},
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


@numba.njit(numba.void(READONLY_F32_2D, WRITABLE_F32_1D), cache=True)
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


@numba.njit(numba.void(READONLY_F32_2D, numba.int64, WRITABLE_F32_1D), parallel=True, cache=True)
def _fill_pdist_cos_parallel(vectors: NDArray[np.float32], block_width: np.int64, out: NDArray[np.float32]) -> None:
    """Write condensed cosine distances of pre-normalized rows into `out`, in parallel."""
    n = vectors.shape[0]
    for j_block in range(0, n, block_width):
        j_end = min(j_block + block_width, n)
        for i in numba.prange(j_end):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
            base = np.int64(i) * n - (np.int64(i) * (i + 1)) // 2 - i - 1
            for j in range(max(j_block, i + 1), j_end):
                out[base + j] = np.float32(0.5 * _l2sq_pair(vectors, np.int32(i), np.int32(j)))
