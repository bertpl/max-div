"""Every distance the package produces is computed by one of the pairwise distance functions here.

The builds in `_build` and the on-demand reads in `_store` all go through these, which is what
keeps stored and on-demand values bit-equal.  Every function reads the array that `DistanceMetric.preprocess`
returns, which is either a preprocessed array for those distance metrics that need it, or the original
array of vectors for those that don't.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit

from ._distance_metric import (
    METRIC_KIND_ALONG_AXIS,
    METRIC_KIND_COS,
    METRIC_KIND_GEOMEAN,
    METRIC_KIND_L1,
    METRIC_KIND_L2,
    METRIC_KIND_L2S,
    METRIC_KIND_LINF,
    METRIC_KIND_LMINUSINF,
    METRIC_KIND_MARGINALS_AND_JOINT,
    METRIC_KIND_MINKOWSKI,
    METRIC_KIND_MINKOWSKI_P0125,
    METRIC_KIND_MINKOWSKI_P025,
    METRIC_KIND_MINKOWSKI_P025_POWERED,
    METRIC_KIND_MINKOWSKI_P05,
    METRIC_KIND_MINKOWSKI_P05_POWERED,
    METRIC_KIND_MINKOWSKI_POWERED,
)

# =================================================================================================
#  Pairwise distance functions
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


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _l1_distance(vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger) -> np.float64:
    """Return the L1 (Manhattan) distance between vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c]))
    return acc


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _l2sq_distance(vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger) -> np.float64:
    """Return the squared L2 (Euclidean) distance between vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        diff = np.float64(vectors[i, c]) - np.float64(vectors[j, c])
        acc += diff * diff
    return acc


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _linf_distance(vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger) -> np.float64:
    """Return the Linf (Chebyshev) distance between vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        diff = abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c]))
        if diff > acc:
            acc = diff
    return acc


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _lminusinf_distance(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger
) -> np.float64:
    """Return the L-∞ distance between vectors i and j: the smallest absolute coordinate difference."""
    acc = np.float64(np.inf)
    for c in range(vectors.shape[1]):
        diff = abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c]))
        if diff < acc:
            acc = diff
    return acc


@lazy_njit(
    "float64(float32[:, ::1], int64, int64, float64)", inline="always", cache=True, fastmath={"reassoc", "contract"}
)
def _marginals_and_joint_distance(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger, joint_scale: np.float64
) -> np.float64:
    """Return the marginals-and-joint distance of vectors i and j, defined on `DistanceMetric.marginals_and_joint`.

    The distance is computed in float64; for a far pair in a high dimension the joint term may overflow
    to +inf, and the minimum then returns the smallest gap.
    """
    # --- smallest coordinate gap (L-∞) ---------
    smallest_gap = _lminusinf_distance(vectors, i, j)

    # --- joint term (L2 to the power d) ---------
    # the metric is meant for small d, so the common dimensions skip the general power
    squared_l2 = _l2sq_distance(vectors, i, j)
    n_dims = vectors.shape[1]
    if n_dims == 2:
        joint_term = squared_l2
    elif n_dims == 3:
        joint_term = squared_l2 * np.sqrt(squared_l2)
    elif n_dims == 4:
        joint_term = squared_l2 * squared_l2
    else:
        joint_term = squared_l2 ** (0.5 * n_dims)

    # --- minimum --------------------------------
    return min(smallest_gap, joint_scale * joint_term)


@lazy_njit(
    "float64(float32[:, ::1], int64, int64, float64)", inline="always", cache=True, fastmath={"reassoc", "contract"}
)
def _minkowski_distance_powered(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger, p: np.float64
) -> np.float64:
    """Return ``sum_c |x_c - y_c|^p`` for vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c])) ** p
    return acc


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _minkowski_distance_powered_p05(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger
) -> np.float64:
    """Return ``sum_c sqrt(|x_c - y_c|)`` for vectors i and j, accumulated in float64."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += np.sqrt(abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c])))
    return acc


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _minkowski_distance_powered_p025(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger
) -> np.float64:
    """Return ``sum_c |x_c - y_c|^0.25`` for vectors i and j, via two square roots per term."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += np.sqrt(np.sqrt(abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c]))))
    return acc


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _minkowski_distance_powered_p0125(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger
) -> np.float64:
    """Return ``sum_c |x_c - y_c|^0.125`` for vectors i and j, via three square roots per term."""
    acc = np.float64(0.0)
    for c in range(vectors.shape[1]):
        acc += np.sqrt(np.sqrt(np.sqrt(abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c])))))
    return acc


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True, fastmath={"reassoc", "contract"})
def _geomean_distance(vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger) -> np.float64:
    """Return the geometric mean of the per-dimension absolute differences of vectors i and j.

    The function sums logarithms and exponentiates once, so the product cannot leave float64's
    range at any dimension count.  A zero difference contributes ``log(0) = -inf`` and the final
    ``exp`` turns that into exactly zero; the fastmath subset above keeps infinities intact, so no
    branch is needed in the loop (measured at 4-8 % of its cost).
    """
    log_sum = np.float64(0.0)
    d = vectors.shape[1]
    for c in range(d):
        log_sum += np.log(abs(np.float64(vectors[i, c]) - np.float64(vectors[j, c])))
    return np.exp(log_sum / d)


@lazy_njit("float64(float32[:, ::1], int64, int64)", inline="always", cache=True)
def _along_axis_distance(
    vectors: NDArray[np.float32], i: int | np.signedinteger, j: int | np.signedinteger
) -> np.float64:
    """Return the absolute difference of column 0 of vectors i and j, the coordinate that preprocessing kept."""
    return abs(np.float64(vectors[i, 0]) - np.float64(vectors[j, 0]))


@lazy_njit(
    numba.float32(numba.float32[:, ::1], numba.int32, numba.float64, numba.int32, numba.int32),
    inline="always",
    cache=True,
)
def _pairwise_distance(  # noqa: C901 -- flat dispatch, one arm per kind: complexity here is roster size, not tangledness
    vectors: NDArray[np.float32],
    metric_kind: np.int32,
    metric_float_param: np.float64,
    i: np.int32,
    j: np.int32,
) -> np.float32:
    """Compute the distance between vectors i and j, per the given metric selector.

    `metric_float_param` is the metric's `DistanceMetric.float_param`: the power `p` of a
    generic Minkowski kind, or the `joint_scale` of marginals-and-joint.

    The selector is loop-invariant in every calling loop, so the branch order is not
    performance-relevant.  The specialized Minkowski kinds apply the outer root as repeated
    squarings.
    """
    if metric_kind == METRIC_KIND_L1:
        return np.float32(_l1_distance(vectors, i, j))
    if metric_kind == METRIC_KIND_L2:
        return np.float32(np.sqrt(_l2sq_distance(vectors, i, j)))
    if metric_kind == METRIC_KIND_L2S:
        return np.float32(_l2sq_distance(vectors, i, j))
    if metric_kind == METRIC_KIND_LINF:
        return np.float32(_linf_distance(vectors, i, j))
    if metric_kind == METRIC_KIND_LMINUSINF:
        return np.float32(_lminusinf_distance(vectors, i, j))
    if metric_kind == METRIC_KIND_COS:
        return np.float32(0.5 * _l2sq_distance(vectors, i, j))  # cosine: rows are pre-normalized
    if metric_kind == METRIC_KIND_GEOMEAN:
        return np.float32(_geomean_distance(vectors, i, j))
    if metric_kind == METRIC_KIND_ALONG_AXIS:
        return np.float32(_along_axis_distance(vectors, i, j))  # along axis: the array holds that one coordinate
    if metric_kind == METRIC_KIND_MARGINALS_AND_JOINT:
        return np.float32(_marginals_and_joint_distance(vectors, i, j, metric_float_param))
    if metric_kind == METRIC_KIND_MINKOWSKI:
        return np.float32(_minkowski_distance_powered(vectors, i, j, metric_float_param) ** (1.0 / metric_float_param))
    if metric_kind == METRIC_KIND_MINKOWSKI_POWERED:
        return np.float32(_minkowski_distance_powered(vectors, i, j, metric_float_param))
    if metric_kind == METRIC_KIND_MINKOWSKI_P05:
        acc = _minkowski_distance_powered_p05(vectors, i, j)
        return np.float32(acc * acc)
    if metric_kind == METRIC_KIND_MINKOWSKI_P05_POWERED:
        return np.float32(_minkowski_distance_powered_p05(vectors, i, j))
    if metric_kind == METRIC_KIND_MINKOWSKI_P025:
        acc = _minkowski_distance_powered_p025(vectors, i, j)
        squared = acc * acc
        return np.float32(squared * squared)
    if metric_kind == METRIC_KIND_MINKOWSKI_P025_POWERED:
        return np.float32(_minkowski_distance_powered_p025(vectors, i, j))
    if metric_kind == METRIC_KIND_MINKOWSKI_P0125:
        acc = _minkowski_distance_powered_p0125(vectors, i, j)
        squared = acc * acc
        fourth = squared * squared
        return np.float32(fourth * fourth)
    else:
        # only MINKOWSKI_P0125_POWERED remains
        return np.float32(_minkowski_distance_powered_p0125(vectors, i, j))
