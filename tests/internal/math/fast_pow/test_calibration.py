import pytest

from max_div.internal.math.fast_pow._calibration import calibrate_fast_pow
from max_div.internal.math.fast_pow._fast_pow import _D_E0 as d0_actual
from max_div.internal.math.fast_pow._fast_pow import _D_E1 as d1_actual
from max_div.internal.math.fast_pow._fast_pow import _D_E2 as d2_actual
from max_div.internal.math.fast_pow._fast_pow import _D_L0 as c0_actual
from max_div.internal.math.fast_pow._fast_pow import _D_L1 as c1_actual
from max_div.internal.math.fast_pow._fast_pow import _D_L2 as c2_actual


@pytest.mark.parametrize("start_from_current", [False, True])
def test_calibrate_fast_pow(start_from_current: bool):
    # --- arrange -----------------------------------------
    n_data = 100
    acc = 1e-4
    n_evals = 10_000

    # --- act ---------------------------------------------
    (c0, c1, c2), (d0, d1, d2) = calibrate_fast_pow(n_data, acc, n_evals, start_from_current)

    # --- assert ------------------------------------------

    if start_from_current:
        tol = 0.05
    else:
        tol = 0.25

    # fast_log coefficients
    assert abs(c0 - c0_actual) <= tol
    assert abs(c1 - c1_actual) <= tol
    assert abs(c2 - c2_actual) <= tol

    # fast_exp coefficients
    assert abs(d0 - d0_actual) <= tol
    assert abs(d1 - d1_actual) <= tol
    assert abs(d2 - d2_actual) <= tol
