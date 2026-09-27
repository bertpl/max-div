"""Compute the geometric mean of a float32 vector, or the weighted one of each matrix row, as exp of the mean log.

Every function here requires at least one entry per reduced vector. `geomean_f32` and
`weighted_geomean_per_row_f32` accept zero and +inf:

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


# Every function here uses the same fastmath subset as the pair-distance functions in
# `_distance/_metric/_pair.py`. The subset omits the `ninf` flag, which would let the compiler assume
# no infinities, so in `geomean_f32` and `weighted_geomean_per_row_f32` a +inf entry stays +inf through the sum
# (`fast_geomean_f32` does not preserve it: its log and exp are approximations).
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


@njit("void(float32[:, ::1], float32[::1], float32[::1])", fastmath={"reassoc", "contract"}, cache=True)
def weighted_geomean_per_row_f32(
    rows: NDArray[np.float32], weights: NDArray[np.float32], out: NDArray[np.float32]
) -> None:
    """Write the weighted geometric mean of each row of `rows` into `out`: each entry raised to its column's weight.

    The mean of row i is (prod_j rows[i, j] ** weights[j]) ** (1 / sum(weights)). `rows` is a
    C-contiguous (n_rows, n_cols) array with n_cols at least one, `weights` holds one positive weight
    per column, and `out` has length n_rows. With every weight 1, each entry of `out` is bit for bit
    the `geomean_f32` of its row: the log sum accumulates in float32 in the same order, and the
    division by the weight sum runs in the precision of `geomean_f32`'s division by the entry count.
    """
    n_rows, n_cols = rows.shape
    weight_sum = 0.0  # float64
    for j in range(n_cols):
        weight_sum += weights[j]
    for i in range(n_rows):
        log_sum = np.float32(0.0)
        for j in range(n_cols):
            log_sum += weights[j] * np.log(rows[i, j])
        out[i] = np.exp(log_sum / weight_sum)
