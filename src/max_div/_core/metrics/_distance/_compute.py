"""Methods for computing pair-wise distances between vectors, along the lines of scipy's pdist."""

import numba
import numpy as np
from numpy.typing import NDArray

from ._enum import DistanceMetric


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
    match metric:
        case DistanceMetric.L1_MANHATTAN:
            _pdist_l1(vectors, out)
        case DistanceMetric.L2_EUCLIDEAN:
            _pdist_l2(vectors, out)
        case DistanceMetric.L2S_EUCLIDEAN_SQUARED:
            _pdist_l2s(vectors, out)
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
@numba.njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True)
def _l1_pair(vectors: NDArray[np.float32], i: int, j: int) -> np.float64:
    """Return the L1 (Manhattan) distance between vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c]))
    return acc


@numba.njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True)
def _l2sq_pair(vectors: NDArray[np.float32], i: int, j: int) -> np.float64:
    """Return the squared L2 (Euclidean) distance between vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        diff = np.float64(vectors[i, c]) - np.float64(vectors[j, c])
        acc += diff * diff
    return acc


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True)
def _pdist_l1(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed L1 distances of `vectors` into pre-allocated `out`, in condensed i<j order."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(_l1_pair(vectors, i, j))
            idx += 1


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True)
def _pdist_l2(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed L2 distances of `vectors` into pre-allocated `out`, in condensed i<j order."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(np.sqrt(_l2sq_pair(vectors, i, j)))
            idx += 1


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True)
def _pdist_l2s(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed squared-L2 distances of `vectors` into pre-allocated `out`, in condensed i<j order."""
    n = vectors.shape[0]
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(_l2sq_pair(vectors, i, j))
            idx += 1


@numba.njit("void(float32[:, ::1], float32[::1])", cache=True)
def _pdist_cos(vectors: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """Write the condensed cosine distances of `vectors` into pre-allocated `out`, in condensed i<j order.

    Computed as ``0.5 * ||x^ - y^||^2`` on unit-normalized vectors, which equals ``1 - cos(x, y)``
    algebraically but — unlike the dot-product form — is non-negative by construction and exactly
    0.0 for identical vectors. Rows are normalized once (norm accumulated in float64) into a
    float32 scratch array, so the pair loop reuses the squared-L2 accumulation.

    Vectors must contain no all-zero rows (`validate_cosine_vectors` guards the public entry point).
    """
    n = vectors.shape[0]
    d = vectors.shape[1]
    # --- normalize rows --------------------------
    normalized = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        acc = np.float64(0.0)
        for c in range(d):
            acc += np.float64(vectors[i, c]) * np.float64(vectors[i, c])
        norm = np.sqrt(acc)
        for c in range(d):
            normalized[i, c] = np.float32(np.float64(vectors[i, c]) / norm)
    # --- pairwise distances ----------------------
    idx = np.int64(0)
    for i in range(n):
        for j in range(i + 1, n):
            out[idx] = np.float32(0.5 * _l2sq_pair(normalized, i, j))
            idx += 1
