import math

import numpy as np
import pytest

from max_div.internal.math.fast_log_exp import fast_exp2_f32, fast_exp2_f64, fast_exp_f32, fast_exp_f64


# =================================================================================================
#  fast_exp
# =================================================================================================
def test_fast_exp_f64():
    # --- arrange -----------------------------------------
    max_diff = 0.003
    x_values = np.linspace(-1, +1, 10_000)
    np_expx_values = np.exp(x_values)

    # --- act ---------------------------------------------
    fast_expx_values = np.array([fast_exp_f64(x) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_expx_values - fast_expx_values) / np_expx_values)
    print(diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


@pytest.mark.parametrize("x0", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_fast_exp_f64_continuity(x0: float):
    # --- arrange -----------------------------------------
    dx = 1e-12
    dfdx_true = math.exp(x0)

    # --- act ---------------------------------------------
    fx_min = fast_exp_f64(np.float64(x0 - dx))
    fx_plus = fast_exp_f64(np.float64(x0 + dx))

    # --- assert ------------------------------------------
    assert abs(fx_plus - fx_min) <= 10 * (2 * dx * dfdx_true)


@pytest.mark.parametrize("x0", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_fast_exp_f64_smoothness(x0: float):
    # --- arrange -----------------------------------------
    dx = 1e-12

    # --- act ---------------------------------------------
    fx_min = fast_exp_f64(np.float64(x0 - dx))
    fx = fast_exp_f64(np.float64(x0))
    fx_plus = fast_exp_f64(np.float64(x0 + dx))

    # --- assert ------------------------------------------
    dfdx_plus = (fx_plus - fx) / dx
    dfdx_min = (fx - fx_min) / dx

    assert 0.99 <= (dfdx_plus / dfdx_min) <= 1.01


def test_fast_exp_f32():
    # --- arrange -----------------------------------------
    max_diff = 0.003
    x_values = np.linspace(-1, +1, 10_000, dtype=np.float32)
    np_expx_values = np.exp(x_values)

    # --- act ---------------------------------------------
    fast_expx_values = np.array([fast_exp_f32(x) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_expx_values - fast_expx_values) / np_expx_values)
    print(diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


# =================================================================================================
#  fast_exp2
# =================================================================================================
def test_fast_exp2_f64():
    # --- arrange -----------------------------------------
    max_diff = 0.003
    x_values = np.linspace(-1, +1, 10_000)
    np_exp2x_values = np.exp2(x_values)

    # --- act ---------------------------------------------
    fast_exp2x_values = np.array([fast_exp2_f64(x) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_exp2x_values - fast_exp2x_values) / np_exp2x_values)
    print(diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


def test_fast_exp2_f32():
    # --- arrange -----------------------------------------
    max_diff = 0.003
    x_values = np.linspace(-1, +1, 10_000, dtype=np.float32)
    np_exp2x_values = np.exp2(x_values)

    # --- act ---------------------------------------------
    fast_exp2x_values = np.array([fast_exp2_f32(x) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_exp2x_values - fast_exp2x_values) / np_exp2x_values)
    print(diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"
