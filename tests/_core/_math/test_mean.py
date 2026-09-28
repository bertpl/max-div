import numpy as np
import pytest

from max_div._core._math.mean import weighted_mean_per_row_f32


@pytest.mark.parametrize("n_cols", [1, 2, 3, 7])
def test_weighted_mean_per_row_f32_at_equal_weights_is_numpys_row_mean_bit_for_bit(n_cols: int) -> None:
    """With every weight 1 and fewer than 8 columns, each entry is exactly numpy's float32 mean of its row."""
    # --- arrange ----------------------
    rows = np.random.default_rng(42).uniform(0.001, 10.0, size=(1000, n_cols)).astype(np.float32)
    out = np.empty(1000, dtype=np.float32)

    # --- act --------------------------
    weighted_mean_per_row_f32(rows, np.ones(n_cols, dtype=np.float32), out)

    # --- assert -----------------------
    np.testing.assert_array_equal(out, rows.mean(axis=1, dtype=np.float32))


def test_weighted_mean_per_row_f32_weights_each_column() -> None:
    """Each row's mean is the weighted sum of its entries divided by the weight sum."""
    # --- arrange ----------------------
    rows = np.array([[2.0, 8.0], [3.0, 0.0]], dtype=np.float32)
    weights = np.array([3.0, 1.0], dtype=np.float32)
    out = np.empty(2, dtype=np.float32)

    # --- act --------------------------
    weighted_mean_per_row_f32(rows, weights, out)

    # --- assert -----------------------
    np.testing.assert_allclose(out, [(3.0 * 2.0 + 8.0) / 4.0, (3.0 * 3.0) / 4.0], rtol=1e-6)
