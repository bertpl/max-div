import numpy as np

from max_div.internal.math.fast_exp import fast_exp2_f32, fast_exp2_f64, fast_exp_f32, fast_exp_f64


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
