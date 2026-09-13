"""Geometric mean of a float32 vector, exact and approximate, as the log-sum-exp of its entries.

Both functions take at least one entry; the caller guarantees that, because an empty input divides
by zero. The exact one accepts zero and +inf: a zero entry makes the mean zero, a +inf entry makes
it +inf, and an input holding both gives nan, since their logs cancel to an undefined sum. The
approximate one is defined for positive normal floats only, where the fast log is: a zero entry
gives a value near zero, not zero, and a +inf entry gives a large finite value.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .fast_log_exp import fast_exp2_f32, fast_log2_f32


# The fastmath subset matches the pair-distance functions in the distance package, so a +inf entry
# keeps its meaning through the reduction.
@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def geomean(values: NDArray[np.float32]) -> np.float32:
    """Geometric mean of the entries."""
    log_sum = np.float32(0.0)
    n = values.shape[0]
    for i in range(n):
        log_sum += np.log(values[i])
    return np.exp(log_sum / n)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def approx_geomean(values: NDArray[np.float32]) -> np.float32:
    """Approximate geometric mean of the entries, through the fast base-2 log and exp."""
    log_sum = np.float32(0.0)
    n = values.shape[0]
    for i in range(n):
        log_sum += fast_log2_f32(values[i])
    return fast_exp2_f32(log_sum / n)
