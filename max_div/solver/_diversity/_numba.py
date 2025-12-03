import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div.internal.math.fast_exp import fast_exp2_f32
from max_div.internal.math.fast_log import fast_log2_f32


@njit("float32(float32[::1])", fastmath=True, inline="always")
def min_separation(sep: NDArray[np.float32]) -> np.float32:
    """Minimum separation of all selected vectors."""
    return np.min(sep)


@njit("float32(float32[::1])", fastmath=True, inline="always")
def mean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Arithmetic mean separation of all selected vectors."""
    return np.mean(sep)


@njit("float32(float32[::1])", fastmath=True, inline="always")
def geomean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Geometric mean separation of all selected vectors."""
    log_sum = np.float32(0.0)
    n = sep.shape[0]
    for i in range(n):
        if sep[i] == 0.0:
            return np.float32(0.0)
        else:
            log_sum += np.log(sep[i])
    return np.exp(log_sum / n)


@njit("float32(float32[::1])", fastmath=True, inline="always")
def approx_geomean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Approximate geometric mean separation of all selected vectors."""
    log_sum = np.float32(0.0)
    n = sep.shape[0]
    for i in range(n):
        if sep[i] == 0.0:
            return np.float32(0.0)
        else:
            log_sum += fast_log2_f32(sep[i])
    return fast_exp2_f32(log_sum / n)
