"""Compute the weighted arithmetic mean of each row of a float32 matrix."""

import numpy as np
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit


# No fastmath flags: `reassoc` would let the compiler reorder each row's sum, and only the in-order
# sum matches numpy's float32 row mean bit for bit at equal weights.
@lazy_njit("void(float32[:, ::1], float32[::1], float32[::1])", cache=True)
def weighted_mean_per_row_f32(
    rows: NDArray[np.float32], weights: NDArray[np.float32], out: NDArray[np.float32]
) -> None:
    """Write the weighted arithmetic mean of each row of `rows` into `out`.

    The mean of row i is sum_j weights[j] * rows[i, j] / sum(weights). Each row sums in column order
    in float32, and the division runs in float32.

    With every weight 1 and fewer than 8 columns, each entry is bit for bit numpy's float32 mean of
    its row, since numpy also sums fewer than 8 values in order.

    Args:
        rows: a C-contiguous (n_rows, n_cols) array with n_cols at least one.
        weights: one positive weight per column.
        out: the output array, of length n_rows.
    """
    n_rows, n_cols = rows.shape
    weight_sum = np.float32(0.0)
    has_unit_weights = True
    for j in range(n_cols):
        weight_sum += weights[j]
        has_unit_weights = has_unit_weights and weights[j] == 1.0

    # the unit-weight branch skips one multiply per entry; when every weight is 1, both branches give
    # the same bits
    if has_unit_weights:
        for i in range(n_rows):
            row_sum = np.float32(0.0)
            for j in range(n_cols):
                row_sum += rows[i, j]
            out[i] = row_sum / weight_sum
    else:
        for i in range(n_rows):
            weighted_sum = np.float32(0.0)
            for j in range(n_cols):
                weighted_sum += weights[j] * rows[i, j]
            out[i] = weighted_sum / weight_sum
