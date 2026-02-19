import numpy as np
import pytest

from max_div._core._math.fast_log_exp import fast_log2_f32, fast_log2_f64, fast_log_f32, fast_log_f64


# =================================================================================================
#  fast_log
# =================================================================================================
def test_fast_log_f64():
    # --- arrange -----------------------------------------
    max_diff = 0.006
    x_values = np.logspace(-1, +4, 10_000)
    np_logx_values = np.log(x_values)

    # --- act ---------------------------------------------
    fast_logx_values = np.array([fast_log_f64(x) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_logx_values - fast_logx_values))
    print(diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


@pytest.mark.parametrize("x0", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_fast_log_f64_continuity(x0: float):
    # --- arrange -----------------------------------------
    dx = 1e-12
    dfdx_true = 1 / x0

    # --- act ---------------------------------------------
    fx_min = fast_log_f64(np.float64(x0 - dx))
    fx_plus = fast_log_f64(np.float64(x0 + dx))

    # --- assert ------------------------------------------
    assert abs(fx_plus - fx_min) <= 10 * (2 * dx * dfdx_true)


@pytest.mark.parametrize("x0", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_fast_log_f64_smoothness(x0: float):
    # --- arrange -----------------------------------------
    dx = 1e-12

    # --- act ---------------------------------------------
    fx_min = fast_log_f64(np.float64(x0 - dx))
    fx = fast_log_f64(np.float64(x0))
    fx_plus = fast_log_f64(np.float64(x0 + dx))

    # --- assert ------------------------------------------
    dfdx_plus = (fx_plus - fx) / dx
    dfdx_min = (fx - fx_min) / dx

    assert 0.99 <= (dfdx_plus / dfdx_min) <= 1.01


def test_fast_log_f32():
    # --- arrange -----------------------------------------
    max_diff = 0.006
    x_values = np.logspace(-1, +4, 10_000, dtype=np.float32)
    np_logx_values = np.log(x_values)

    # --- act ---------------------------------------------
    fast_logx_values = np.array([fast_log_f32(x) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_logx_values - fast_logx_values))
    print(diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


# =================================================================================================
#  fast_log2
# =================================================================================================
def test_fast_log2_f64():
    # --- arrange -----------------------------------------
    max_diff = 0.008
    x_values = np.logspace(-1, +4, 10_000)
    np_log2x_values = np.log2(x_values)

    # --- act ---------------------------------------------
    fast_log2x_values = np.array([fast_log2_f64(x) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_log2x_values - fast_log2x_values))
    print(diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


def test_fast_log2_f32():
    # --- arrange -----------------------------------------
    max_diff = 0.008
    x_values = np.logspace(-1, +4, 10_000, dtype=np.float32)
    np_log2x_values = np.log2(x_values)

    # --- act ---------------------------------------------
    fast_log2x_values = np.array([fast_log2_f32(x) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_log2x_values - fast_log2x_values))
    print(diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"
