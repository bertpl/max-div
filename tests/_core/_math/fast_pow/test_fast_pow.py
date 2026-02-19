import os

import numba
import numpy as np

from max_div._core._math.fast_pow import construct_calibration_data, fast_pow_f32


def test_fast_pow_f32_accuracy():
    # --- arrange -----------------------------------------
    x_arr, t_arr, xt_exact_arr = construct_calibration_data(199)
    max_abs_tol = 0.08

    # --- act ---------------------------------------------
    xt_approx_arr = np.array(
        [fast_pow_f32(np.float32(x), np.float32(t)) for x, t in zip(x_arr, t_arr)],
        dtype=np.float32,
    )

    # --- assert ------------------------------------------
    assert max(abs(xt_exact_arr - xt_approx_arr)) <= max_abs_tol


def test_fast_pow_f32_llvm_output():
    # --- arrange -----------------------------------------
    fast_pow_f32(np.float32(0.5), np.float32(1.2))  # trigger compilation

    # --- act ---------------------------------------------
    jit_disabled = os.environ.get("NUMBA_DISABLE_JIT", "0")
    if jit_disabled not in {"1", "true", "True"}:
        print(fast_pow_f32.inspect_llvm((numba.float32, numba.float32)))  # dump LLVM IR
