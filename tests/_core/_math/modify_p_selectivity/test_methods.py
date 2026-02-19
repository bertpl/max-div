from typing import Callable

import numpy as np
import pytest

from max_div._core._math.modify_p_selectivity._methods import (
    _max_selective,
    _power_exact,
    _power_fast_log2_exp2,
    _power_fast_pow,
    _pwl_2_segment,
    _uniform,
)


# =================================================================================================
#  Boundary methods
# =================================================================================================
def test_modify_p_methods_uniform():
    # --- arrange -----------------------------------------
    p = np.array([0.0, 0.123, 0.002, 0.345, 0.0], dtype=np.float32)

    # --- act ---------------------------------------------
    _uniform(p)

    # --- assert ------------------------------------------
    assert min(p) == max(p) == np.float32(1.0)


def test_modify_p_methods_max_selective():
    # --- arrange -----------------------------------------
    p = np.array([0.0, 0.123, 0.002, 1.0, 0.0, 1.0], dtype=np.float32)

    # --- act ---------------------------------------------
    _max_selective(p)

    # --- assert ------------------------------------------
    assert np.array_equal(
        p,
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )


# =================================================================================================
#  Regular methods
# =================================================================================================
METHODS_AND_TOLERANCES = [
    (_power_exact, 1e-6),
    (_power_fast_log2_exp2, 1e-2),
    (_power_fast_pow, 1e-2),
    (_pwl_2_segment, 1e-1),
]


@pytest.mark.parametrize("fun,tol", METHODS_AND_TOLERANCES)
def test_modify_p_methods_accuracy(fun: Callable, tol: float):
    # --- arrange -----------------------------------------
    n = 50
    p = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    modifiers = np.linspace(-0.9, 0.9, num=n, dtype=np.float32)

    p_out_expected = [np.array([p[i] ** ((1 + m) / (1 - m)) for i in range(n)], dtype=np.float32) for m in modifiers]

    # --- act ---------------------------------------------
    e_tot = 0.0
    for modifier, expected_result in zip(modifiers, p_out_expected):
        p_out = p.copy()
        fun(p_out, modifier)
        e_tot += sum(abs(p_out - expected_result))

    e_tot /= n * n

    # --- assert ------------------------------------------
    assert e_tot <= tol
