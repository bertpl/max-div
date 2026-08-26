import pytest

from max_div._core._cli.benchmarks._helpers.speed_scaling import SpeedParam


def test_speed_param_endpoints():
    """`at()` returns exactly the declared values at the speed endpoints, on both scales."""
    # --- arrange ----------------------
    log_param = SpeedParam(slow=600.0, fast=1e-3)
    linear_param = SpeedParam(slow=25, fast=1, scale="linear")

    # --- act / assert -----------------
    assert log_param.at(0.0) == pytest.approx(600.0)
    assert log_param.at(1.0) == pytest.approx(1e-3)
    assert linear_param.at(0.0) == 25
    assert linear_param.at(1.0) == 1


def test_speed_param_log_interpolation():
    """Log-scale interpolation is geometric: mid-speed lands on the geometric mean."""
    # --- act --------------------------
    value = SpeedParam(slow=100.0, fast=1.0).at(0.5)

    # --- assert -----------------------
    assert value == pytest.approx(10.0)


def test_speed_param_linear_interpolation():
    """Linear-scale interpolation is arithmetic: mid-speed lands on the arithmetic mean."""
    # --- act --------------------------
    value = SpeedParam(slow=8.0, fast=2.0, scale="linear").at(0.5)

    # --- assert -----------------------
    assert value == pytest.approx(5.0)


def test_speed_param_at_int_rounds():
    """`at_int()` rounds to the nearest integer and returns an int."""
    # --- act --------------------------
    value = SpeedParam(slow=50, fast=2).at_int(0.5)

    # --- assert -----------------------
    assert isinstance(value, int)
    assert value == round(50 * (2 / 50) ** 0.5)


def test_speed_param_log_rejects_non_positive():
    """Log scale requires strictly positive endpoint values."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="strictly positive"):
        SpeedParam(slow=0.0, fast=1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        SpeedParam(slow=100.0, fast=-1.0)


def test_speed_param_rejects_positional_construction():
    """Construction is keyword-only, so positional endpoint values raise."""
    # --- act / assert -----------------
    with pytest.raises(TypeError):
        SpeedParam(600.0, 1e-3)  # type: ignore[misc]
