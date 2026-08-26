import pytest

from max_div._core._cli.bm_speed import SpeedParam


def test_speed_param_endpoints():
    """`at()` returns exactly the declared values at the speed endpoints, on both scales."""
    # --- arrange ----------------------
    log_param = SpeedParam(600.0, 1e-3)
    linear_param = SpeedParam(25, 1, scale="linear")

    # --- act / assert -----------------
    assert log_param.at(0.0) == pytest.approx(600.0)
    assert log_param.at(1.0) == pytest.approx(1e-3)
    assert linear_param.at(0.0) == 25
    assert linear_param.at(1.0) == 1


def test_speed_param_log_interpolation():
    """Log-scale interpolation is geometric: mid-speed lands on the geometric mean."""
    # --- act --------------------------
    value = SpeedParam(100.0, 1.0).at(0.5)

    # --- assert -----------------------
    assert value == pytest.approx(10.0)


def test_speed_param_linear_interpolation():
    """Linear-scale interpolation is arithmetic: mid-speed lands on the arithmetic mean."""
    # --- act --------------------------
    value = SpeedParam(8.0, 2.0, scale="linear").at(0.5)

    # --- assert -----------------------
    assert value == pytest.approx(5.0)


def test_speed_param_at_int_rounds():
    """`at_int()` rounds to the nearest integer and returns an int."""
    # --- act --------------------------
    value = SpeedParam(50, 2).at_int(0.5)

    # --- assert -----------------------
    assert isinstance(value, int)
    assert value == round(50 * (2 / 50) ** 0.5)


def test_speed_param_log_rejects_non_positive():
    """Log scale requires strictly positive endpoint values."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="strictly positive"):
        SpeedParam(0.0, 1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        SpeedParam(100.0, -1.0)
