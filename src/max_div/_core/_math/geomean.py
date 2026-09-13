"""Compute the geometric mean of a float32 vector, exactly or approximately, as the exponential of the mean log.

Both functions require at least one entry. `geomean_f32` accepts zero and +inf:

- a zero entry makes the mean zero;
- a +inf entry makes it +inf;
- an input holding both gives nan, since log 0 = -inf and log +inf = +inf sum to nan.

`fast_geomean_f32` is defined only for positive normal floats, the domain of `fast_log2_f32`: a zero
entry gives a value near zero, not zero, and a +inf entry gives a large finite value.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .fast_log_exp import fast_exp2_f32, fast_log2_f32


# Both functions use the same fastmath subset as the pair-distance functions in
# `_distance/_metric/_pair.py`. The subset omits the `ninf` flag, which would let the compiler assume
# no infinities, so in `geomean_f32` a +inf entry stays +inf through the sum.
@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def geomean_f32(values: NDArray[np.float32]) -> np.float32:
    """Return the geometric mean of the entries.

    `values` must hold at least one entry; an input holding both a zero and a +inf entry gives nan.
    """
    log_sum = np.float32(0.0)
    n = values.shape[0]
    for i in range(n):
        log_sum += np.log(values[i])
    return np.exp(log_sum / n)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def fast_geomean_f32(values: NDArray[np.float32]) -> np.float32:
    """Return an approximate geometric mean of the entries, computed with the fast base-2 log and exp.

    `values` must hold at least one entry, and every entry must be a positive normal float. The result
    is within about one percent of the exact mean; the bound follows from the errors of
    `fast_log2_f32` and `fast_exp2_f32`.
    """
    log_sum = np.float32(0.0)
    n = values.shape[0]
    for i in range(n):
        log_sum += fast_log2_f32(values[i])
    return fast_exp2_f32(log_sum / n)
