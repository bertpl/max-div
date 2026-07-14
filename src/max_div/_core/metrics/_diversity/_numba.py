import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div._core._math.fast_log_exp import fast_exp2_f32, fast_log2_f32


@njit("float32(float32[::1])", fastmath=True, inline="always")
def min_separation(sep: NDArray[np.float32]) -> np.float32:
    """Minimum separation of all selected vectors."""
    n = sep.shape[0]
    min_value = np.float32(np.inf)
    for i in range(n):
        min_value = min(min_value, sep[i])
    return min_value


@njit("float32(float32[::1])", fastmath=True, inline="always")
def mean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Arithmetic mean separation of all selected vectors."""
    return np.mean(sep)


@njit("float32(float32[::1])", fastmath=True, inline="always")
def mean_pairwise_distance(mean_dists: NDArray[np.float32]) -> np.float32:
    """Mean pairwise distance among all selected vectors, from their mean-distance contribution values.

    Each selected vector's contribution is its mean distance to the other selected vectors; averaging
    those per-vector means over the selection yields exactly the mean over all selected pairs.
    """
    return np.mean(mean_dists)


@njit("float32(float32[::1])", fastmath=True, inline="always")
def geomean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Geometric mean separation of all selected vectors."""
    log_sum = np.float32(0.0)
    n = sep.shape[0]
    for i in range(n):
        log_sum += np.log(sep[i])
    return np.exp(log_sum / n)


@njit("float32(float32[::1])", fastmath=True, inline="always")
def approx_geomean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Approximate geometric mean separation of all selected vectors."""
    log_sum = np.float32(0.0)
    n = sep.shape[0]
    for i in range(n):
        log_sum += fast_log2_f32(sep[i])
    return fast_exp2_f32(log_sum / n)


@njit("float32(float32[::1])", fastmath=True, inline="always")
def non_zero_separation_frac(sep: NDArray[np.float32]) -> np.float32:
    n = sep.shape[0]
    n_non_zero = np.int32(0)
    for i in range(n):
        if sep[i] != 0.0:
            n_non_zero += 1
    return np.float32(n_non_zero) / np.float32(n)
