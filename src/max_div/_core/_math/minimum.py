"""Compute the weighted minimum of each row of a float32 matrix."""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit("void(float32[:, ::1], float32[::1], float32[::1])", cache=True)
def weighted_minimum_per_row_f32(
    rows: NDArray[np.float32], weights: NDArray[np.float32], out: NDArray[np.float32]
) -> None:
    """Write the weighted minimum of each row of `rows` into `out`.

    The minimum of row i is min_j weights[j] * rows[i, j], each product in float32.

    Args:
        rows: a C-contiguous (n_rows, n_cols) array with n_cols at least one.
        weights: one positive weight per column.
        out: the output array, of length n_rows.
    """
    n_rows, n_cols = rows.shape
    has_unit_weights = True
    for j in range(n_cols):
        has_unit_weights = has_unit_weights and weights[j] == 1.0

    # the unit-weight branch skips one multiply per entry; when every weight is 1, both branches give
    # the same bits
    if has_unit_weights:
        for i in range(n_rows):
            smallest = rows[i, 0]
            for j in range(1, n_cols):
                smallest = min(smallest, rows[i, j])
            out[i] = smallest
    else:
        for i in range(n_rows):
            smallest = weights[0] * rows[i, 0]
            for j in range(1, n_cols):
                smallest = min(smallest, weights[j] * rows[i, j])
            out[i] = smallest
