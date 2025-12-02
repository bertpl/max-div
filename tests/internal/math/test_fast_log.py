import numpy as np

from max_div.internal.math.fast_log import fast_log2_f32, fast_log2_f64, fast_log_f32, fast_log_f64


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
