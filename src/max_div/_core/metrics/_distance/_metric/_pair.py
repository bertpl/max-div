"""Every distance the package produces is computed by one of the pair functions here.

The builds in `_build` and the on-demand reads in `_store` all go through these, which is what
keeps stored and on-demand values bit-equal.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from ._distance_metric import (
    METRIC_KIND_COS,
    METRIC_KIND_L1,
    METRIC_KIND_L2,
    METRIC_KIND_L2S,
    METRIC_KIND_LINF,
    METRIC_KIND_MINKOWSKI,
    METRIC_KIND_MINKOWSKI_P025,
    METRIC_KIND_MINKOWSKI_P05,
    METRIC_KIND_MINKOWSKI_P05_POWERED,
    METRIC_KIND_MINKOWSKI_POWERED,
)


def validate_cosine_vectors(vectors: NDArray[np.float32]) -> None:
    """Raise ValueError if any vector is all-zero — cosine distance is undefined for zero vectors.

    Args:
        vectors: (n x d ndarray) A set of n vectors in d dimensions.
    """
    zero_rows = np.flatnonzero(~vectors.any(axis=1))
    if zero_rows.size > 0:
        raise ValueError(
            f"Cosine distance is undefined for zero vectors; found an all-zero vector at row {zero_rows[0]}."
        )


# =================================================================================================
#  Pair functions
# =================================================================================================
# These functions sum one term per dimension.  Adding those terms in a different order gives a
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


@numba.njit(
    "float64(float32[:, ::1], int64, int64, float64)", inline="always", cache=True, fastmath={"reassoc", "contract"}
)
def _minkowski_sum_pair(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger, p: np.float64
) -> np.float64:
    """Return ``sum_c |x_c - y_c|^p`` for vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c])) ** p
    return acc


@numba.njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _minkowski_sum_p05_pair(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger
) -> np.float64:
    """Return ``sum_c sqrt(|x_c - y_c|)`` for vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += np.sqrt(abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c])))
    return acc


@numba.njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _minkowski_sum_p025_pair(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger
) -> np.float64:
    """Return ``sum_c |x_c - y_c|^0.25`` for vectors i and j, via two square roots per term."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += np.sqrt(np.sqrt(abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c]))))
    return acc


@numba.njit(
    numba.float32(numba.float32[:, ::1], numba.int32, numba.float64, numba.int32, numba.int32),
    inline="always",
    cache=True,
)
def _minkowski_pair(
    vectors: NDArray[np.float32], metric_kind: np.int32, metric_p: np.float64, i: np.int32, j: np.int32
) -> np.float32:
    """Compute the Minkowski distance between vectors i and j, per the given Minkowski kind.

    The specialized p = 0.5 / 0.25 kinds apply the outer root as one / two squarings.
    """
    if metric_kind == METRIC_KIND_MINKOWSKI:
        return np.float32(_minkowski_sum_pair(vectors, i, j, metric_p) ** (1.0 / metric_p))
    if metric_kind == METRIC_KIND_MINKOWSKI_POWERED:
        return np.float32(_minkowski_sum_pair(vectors, i, j, metric_p))
    if metric_kind == METRIC_KIND_MINKOWSKI_P05:
        acc = _minkowski_sum_p05_pair(vectors, i, j)
        return np.float32(acc * acc)
    if metric_kind == METRIC_KIND_MINKOWSKI_P05_POWERED:
        return np.float32(_minkowski_sum_p05_pair(vectors, i, j))
    if metric_kind == METRIC_KIND_MINKOWSKI_P025:
        acc = _minkowski_sum_p025_pair(vectors, i, j)
        squared = acc * acc
        return np.float32(squared * squared)
    return np.float32(_minkowski_sum_p025_pair(vectors, i, j))  # MINKOWSKI_P025_POWERED


@numba.njit(
    numba.float32(numba.float32[:, ::1], numba.int32, numba.float64, numba.int32, numba.int32),
    inline="always",
    cache=True,
)
def _metric_pair(
    vectors: NDArray[np.float32], metric_kind: np.int32, metric_p: np.float64, i: np.int32, j: np.int32
) -> np.float32:
    """Compute the distance between vectors i and j, per the given metric selector.

    The Minkowski kinds sit behind the final fall-through, so the original metrics pay no extra
    compares.
    """
    if metric_kind == METRIC_KIND_L1:
        return np.float32(_l1_pair(vectors, i, j))
    if metric_kind == METRIC_KIND_L2:
        return np.float32(np.sqrt(_l2sq_pair(vectors, i, j)))
    if metric_kind == METRIC_KIND_L2S:
        return np.float32(_l2sq_pair(vectors, i, j))
    if metric_kind == METRIC_KIND_LINF:
        return np.float32(_linf_pair(vectors, i, j))
    if metric_kind == METRIC_KIND_COS:
        return np.float32(0.5 * _l2sq_pair(vectors, i, j))  # cosine: rows are pre-normalized
    return _minkowski_pair(vectors, metric_kind, metric_p, i, j)


@numba.njit(numba.float32[:, ::1](numba.types.Array(numba.float32, 2, "C", readonly=True)), cache=True)
def normalize_rows(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a fresh float32 array with each row of `vectors` scaled to unit L2 norm.

    Norms accumulate in float64 and each element narrows to float32 on store, so the result is the
    exact normalization the cosine pair functions operate on.  Rows must not be all-zero
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
