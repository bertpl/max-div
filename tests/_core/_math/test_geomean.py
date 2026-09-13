import numpy as np
import pytest

from max_div._core._math.geomean import fast_geomean_f32, geomean_f32, geomean_per_column_f32

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


def test_geomean_per_column_f32_matches_geomean_f32_per_column() -> None:
    """Each output entry is the geometric mean of that column, including the zero and +inf cases."""
    # --- arrange ----------------------
    values = np.array([[0.1, 2.0, 0.0, 5.0, np.inf], [0.4, 3.0, 1.0, 5.0, 2.0]], dtype=np.float32)
    out = np.empty(5, dtype=np.float32)

    # --- act --------------------------
    geomean_per_column_f32(values, out)

    # --- assert -----------------------
    expected = [geomean_f32(np.ascontiguousarray(values[:, i])) for i in range(5)]
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    assert out[2] == 0.0
    assert np.isinf(out[4])
