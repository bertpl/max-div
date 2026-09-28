import numpy as np
import pytest

from max_div._core._math.geomean import fast_geomean_f32, geomean_f32, weighted_geomean_per_row_f32

# Positive normal floats only, the domain `fast_geomean_f32` is defined on, so both functions can
# share these cases.
_POSITIVE_CASES = [
    ([0.1, 0.4], 0.2),
    ([2.0, 3.0, 4.0], 24.0 ** (1.0 / 3.0)),
    ([1.0, 1.0, 1.0, 1.0], 1.0),
    ([5.0], 5.0),
]


@pytest.mark.parametrize(
    "values, expected",
    [
        *_POSITIVE_CASES,
        ([0.1, 0.0], 0.0),
        ([0.0, 0.0], 0.0),
        ([np.inf], np.inf),
        ([0.1, np.inf], np.inf),
    ],
)
def test_geomean_f32(values: list[float], expected: float) -> None:
    """`geomean_f32` returns the exact geometric mean."""
    # --- arrange ----------------------
    values = np.array(values, dtype=np.float32)

    # --- act --------------------------
    result = geomean_f32(values)

    # --- assert -----------------------
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-6)


# 0.01 is the documented one-percent bound; a zero entry gives about 2^-63, hence the tight tolerance.
@pytest.mark.parametrize(
    "values, expected, tol",
    [
        *[(values, expected, 0.01) for values, expected in _POSITIVE_CASES],
        ([0.1, 0.0], 0.0, 1e-6),
    ],
)
def test_fast_geomean_f32(values: list[float], expected: float, tol: float) -> None:
    """`fast_geomean_f32` is within each case's tolerance of the exact geometric mean."""
    # --- arrange ----------------------
    values = np.array(values, dtype=np.float32)

    # --- act --------------------------
    result = fast_geomean_f32(values)

    # --- assert -----------------------
    assert result == pytest.approx(expected, rel=tol, abs=tol)


def test_weighted_geomean_per_row_f32_at_equal_weights_is_geomean_f32_of_each_row_bit_for_bit() -> None:
    """With every weight 1, each output entry is exactly the `geomean_f32` of its row, zero and +inf rows included."""
    # --- arrange ----------------------
    rng = np.random.default_rng(42)
    values = rng.uniform(0.001, 10.0, size=(1000, 3)).astype(np.float32)
    values[0, 1] = 0.0
    values[1, 2] = np.inf
    out = np.empty(1000, dtype=np.float32)

    # --- act --------------------------
    weighted_geomean_per_row_f32(values, np.ones(3, dtype=np.float32), out)

    # --- assert -----------------------
    expected = np.array([geomean_f32(values[i, :]) for i in range(1000)], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)
    assert out[0] == 0.0
    assert np.isinf(out[1])


def test_weighted_geomean_per_row_f32_raises_each_entry_to_its_columns_weight() -> None:
    """Each row's mean is the product of its entries raised to their weights, to the power 1 / the weight sum."""
    # --- arrange ----------------------
    values = np.array([[2.0, 8.0], [3.0, 0.5], [4.0, 0.0]], dtype=np.float32)
    weights = np.array([2.0, 0.5], dtype=np.float32)
    out = np.empty(3, dtype=np.float32)

    # --- act --------------------------
    weighted_geomean_per_row_f32(values, weights, out)

    # --- assert -----------------------
    expected = (values[:, 0].astype(np.float64) ** 2.0 * values[:, 1].astype(np.float64) ** 0.5) ** (1.0 / 2.5)
    np.testing.assert_allclose(out, expected, rtol=1e-6)
