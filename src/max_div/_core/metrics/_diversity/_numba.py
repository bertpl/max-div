import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div._core._math.geomean import fast_geomean_f32, geomean_f32


# Every reducer here takes the same fastmath subset as the pair-distance functions in
# `_distance/_metric/_pair.py`, for the same reason: `sep` carries +inf for "no selected neighbor yet".
@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def min_separation(sep: NDArray[np.float32]) -> np.float32:
    """Minimum separation of all selected items."""
    n = sep.shape[0]
    min_value = np.float32(np.inf)
    for i in range(n):
        min_value = min(min_value, sep[i])
    return min_value


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def mean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Arithmetic mean separation of all selected items."""
    return np.mean(sep)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def mean_pairwise_distance(mean_dists: NDArray[np.float32]) -> np.float32:
    """Mean pairwise distance among all selected items, from their mean-distance contribution values.

    Each selected item's contribution is its mean distance to the other selected items; averaging
    those per-item means over the selection yields exactly the mean over all selected pairs.
    """
    return np.mean(mean_dists)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def geomean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Geometric mean separation of all selected items."""
    return geomean_f32(sep)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def approx_geomean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Approximate geometric mean separation of all selected items."""
    return fast_geomean_f32(sep)


# `error_model="numpy"` makes a float division by zero yield +inf instead of raising, so the zero and
# +inf limits below need no branch in the loop, which lets it vectorize.
@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, error_model="numpy", inline="always", cache=True)
def harmonic_mean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Harmonic mean separation of all selected items: their count over the sum of their reciprocals.

    Zero and +inf are the limits of that formula, reached through IEEE arithmetic: a zero separation
    puts +inf into the reciprocal sum, so the mean is zero; a +inf separation adds nothing to it, so a
    selection whose separations are all +inf gives +inf.
    """
    n = sep.shape[0]
    reciprocal_sum = np.float32(0.0)
    for i in range(n):
        reciprocal_sum += np.float32(1.0) / sep[i]
    return np.float32(n) / reciprocal_sum


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def non_zero_separation_frac(sep: NDArray[np.float32]) -> np.float32:
    n = sep.shape[0]
    n_non_zero = np.int32(0)
    for i in range(n):
        if sep[i] != 0.0:
            n_non_zero += 1
    return np.float32(n_non_zero) / np.float32(n)
