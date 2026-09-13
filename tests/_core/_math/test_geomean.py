import numpy as np
import pytest

from max_div._core._math.geomean import fast_geomean_f32, geomean_f32

# Both functions share these cases, positive normal floats only; each pair is the input and its
# exact geometric mean.
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
    """`geomean_f32` returns the exact geometric mean for each parametrized input."""
    # --- arrange ----------------------
    values = np.array(values, dtype=np.float32)

    # --- act --------------------------
    result = geomean_f32(values)

    # --- assert -----------------------
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-6)


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
