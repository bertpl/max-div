import numpy as np
import pytest

from max_div._core._math.geomean import approx_geomean, geomean

# Cases shared by both functions: input, exact geometric mean.
_FINITE_CASES = [
    ([0.1, 0.4], 0.2),
    ([2.0, 3.0, 4.0], 24.0 ** (1.0 / 3.0)),
    ([1.0, 1.0, 1.0, 1.0], 1.0),
    ([5.0], 5.0),
]


@pytest.mark.parametrize(
    "values, expected",
    [
        *_FINITE_CASES,
        ([0.1, 0.0], 0.0),
        ([0.0, 0.0], 0.0),
        ([np.inf], np.inf),
        ([0.1, np.inf], np.inf),
    ],
)
def test_geomean(values: list[float], expected: float) -> None:
    """The exact geometric mean holds for finite entries, a single entry, a zero entry and a +inf entry."""
    # --- arrange ----------------------
    values = np.array(values, dtype=np.float32)

    # --- act --------------------------
    result = geomean(values)

    # --- assert -----------------------
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    "values, expected, tol",
    [
        *[(values, expected, 0.01) for values, expected in _FINITE_CASES],
        ([0.1, 0.0], 0.0, 1e-6),
    ],
)
def test_approx_geomean(values: list[float], expected: float, tol: float) -> None:
    """The approximate geometric mean is within a percent for finite entries and near zero for a zero entry."""
    # --- arrange ----------------------
    values = np.array(values, dtype=np.float32)

    # --- act --------------------------
    result = approx_geomean(values)

    # --- assert -----------------------
    assert result == pytest.approx(expected, rel=tol, abs=tol)
