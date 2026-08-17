from max_div._core._math.fast_log_exp._calibration import calibrate_fast_log_exp, exp2_approx, log2_approx
from max_div._core._math.fast_log_exp._fast_exp import _D20 as d0_actual
from max_div._core._math.fast_log_exp._fast_exp import _D21 as d1_actual
from max_div._core._math.fast_log_exp._fast_exp import _D22 as d2_actual
from max_div._core._math.fast_log_exp._fast_log import _D20 as c0_actual
from max_div._core._math.fast_log_exp._fast_log import _D21 as c1_actual
from max_div._core._math.fast_log_exp._fast_log import _D22 as c2_actual


def test_calibrate_fast_log_exp():
    # --- arrange ----------------------
    n = 100
    acc = 1e-2
    n_evals = 1000

    # --- act --------------------------
    (c0, c1, c2), (d0, d1, d2) = calibrate_fast_log_exp(n, acc, n_evals)

    # --- assert -----------------------

    # fast_log coefficients
    assert abs(c0 - c0_actual) <= 2 * acc
    assert abs(c1 - c1_actual) <= 2 * acc
    assert abs(c2 - c2_actual) <= 2 * acc

    # fast_exp coefficients
    assert abs(d0 - d0_actual) <= 2 * acc
    assert abs(d1 - d1_actual) <= 2 * acc
    assert abs(d2 - d2_actual) <= 2 * acc


def test_log2_exp2_approx_edge_cases():
    """Check if local log2_approx, exp2_approx calibration functions accept values beyond the default range."""

    # log2_approx
    _ = log2_approx(-0.1, 1.0, 1.0, 1.0)
    _ = log2_approx(0.4, 1.0, 1.0, 1.0)
    _ = log2_approx(1.1, 1.0, 1.0, 1.0)

    # exp2_approx
    _ = exp2_approx(-0.1, 1.0, 1.0, 1.0)
    _ = exp2_approx(1.1, 1.0, 1.0, 1.0)
