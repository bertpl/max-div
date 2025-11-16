import numpy as np
import pytest

from max_div.internal.math._fast_log import fast_log2_f32_poly, fast_log2_f64_poly, fast_log_f32_poly, fast_log_f64_poly


@pytest.mark.parametrize(
    "degree,max_diff",
    [
        (2, 4e-3),
        (3, 6e-4),
        (4, 7e-5),
        (5, 1.1e-5),
    ],
)
def test_fast_log_f64_poly(degree: int, max_diff: float):
    # --- arrange -----------------------------------------
    x_values = np.logspace(-1, +4, 10_000)
    np_logx_values = np.log(x_values)

    # --- act ---------------------------------------------
    fast_logx_values = np.array([fast_log_f64_poly(x, degree=degree) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_logx_values - fast_logx_values))
    print(degree, diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


@pytest.mark.parametrize(
    "degree,max_diff",
    [
        (2, 6e-3),
        (3, 8e-4),
        (4, 1e-4),
        (5, 2e-5),
    ],
)
def test_fast_log2_f64_poly(degree: int, max_diff: float):
    # --- arrange -----------------------------------------
    x_values = np.logspace(-1, +4, 10_000)
    np_log2x_values = np.log2(x_values)

    # --- act ---------------------------------------------
    fast_log2x_values = np.array([fast_log2_f64_poly(x, degree=degree) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_log2x_values - fast_log2x_values))
    print(degree, diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


@pytest.mark.parametrize(
    "degree,max_diff",
    [
        (2, 4e-3),
        (3, 6e-4),
        (4, 7e-5),
        (5, 1.1e-5),
    ],
)
def test_fast_log_f32_poly(degree: int, max_diff: float):
    # --- arrange -----------------------------------------
    x_values = np.logspace(-1, +4, 10_000, dtype=np.float32)
    np_logx_values = np.log(x_values)

    # --- act ---------------------------------------------
    fast_logx_values = np.array([fast_log_f32_poly(x, degree=degree) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_logx_values - fast_logx_values))
    print(degree, diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"


@pytest.mark.parametrize(
    "degree,max_diff",
    [
        (2, 6e-3),
        (3, 8e-4),
        (4, 1e-4),
        (5, 2e-5),
    ],
)
def test_fast_log2_f32_poly(degree: int, max_diff: float):
    # --- arrange -----------------------------------------
    x_values = np.logspace(-1, +4, 10_000, dtype=np.float32)
    np_log2x_values = np.log2(x_values)

    # --- act ---------------------------------------------
    fast_log2x_values = np.array([fast_log2_f32_poly(x, degree=degree) for x in x_values])

    # --- assert ------------------------------------------
    diff = max(np.abs(np_log2x_values - fast_log2x_values))
    print(degree, diff)

    assert diff <= max_diff, f"Max diff {diff} exceeds allowed {max_diff}"
