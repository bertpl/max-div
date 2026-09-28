import numpy as np

from max_div._core._math.minimum import weighted_minimum_per_row_f32


def test_weighted_minimum_per_row_f32_is_the_smallest_weighted_entry_of_each_row() -> None:
    """Each row's minimum is taken over its entries multiplied by their column's weight, zero and +inf included."""
    # --- arrange ----------------------
    rows = np.array([[2.0, 8.0], [3.0, 0.5], [0.0, 1.0], [np.inf, 4.0], [np.inf, np.inf]], dtype=np.float32)
    weights = np.array([3.0, 1.0], dtype=np.float32)
    out = np.empty(5, dtype=np.float32)

    # --- act --------------------------
    weighted_minimum_per_row_f32(rows, weights, out)

    # --- assert -----------------------
    np.testing.assert_array_equal(out, [6.0, 0.5, 0.0, 4.0, np.inf])


def test_weighted_minimum_per_row_f32_at_unit_weights_is_the_plain_row_minimum() -> None:
    """With every weight 1, each entry is the row's plain minimum, zero and +inf included."""
    # --- arrange ----------------------
    rows = np.array([[2.0, 8.0], [3.0, 0.5], [0.0, 1.0], [np.inf, 4.0], [np.inf, np.inf]], dtype=np.float32)
    out = np.empty(5, dtype=np.float32)

    # --- act --------------------------
    weighted_minimum_per_row_f32(rows, np.ones(2, dtype=np.float32), out)

    # --- assert -----------------------
    np.testing.assert_array_equal(out, rows.min(axis=1))
