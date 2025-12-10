from typing import Callable

import numpy as np
import pytest

from max_div.internal.math.modify_p_selectivity import (
    _max_selective,
    _p_max,
    _uniform,
    modify_p_selectivity_power,
    modify_p_selectivity_pwl2,
)


# =================================================================================================
#  Test Helpers
# =================================================================================================
def test_p_max():
    # --- arrange -----------------------------------------
    p_1 = np.array([0.0, 0.123, 0.002, 0.345, 0.0], dtype=np.float32)
    p_2 = np.array([-1.0, -1.5, -3.0], dtype=np.float32)
    p_3 = np.array([], dtype=np.float32)

    # --- act ---------------------------------------------
    p_max_1 = _p_max(p_1)
    p_max_2 = _p_max(p_2)
    p_max_3 = _p_max(p_3)

    # --- assert ------------------------------------------
    assert p_max_1 == np.float32(0.345)
    assert p_max_2 == np.float32(0.0)
    assert p_max_3 == np.float32(0.0)


def test_uniform():
    # --- arrange -----------------------------------------
    p = np.array([0.0, 0.123, 0.002, 0.345, 0.0], dtype=np.float32)

    # --- act ---------------------------------------------
    u = _uniform(p)

    # --- assert ------------------------------------------
    assert np.allclose(u, np.full(5, 0.345, dtype=np.float32))
    assert u is not p, "Returned array should be a new array, not a reference to the input."


def test_max_selective():
    # --- arrange -----------------------------------------
    p = np.array([0.0, 0.345, 0.002, 0.345, 0.0, 0.344], dtype=np.float32)

    # --- act ---------------------------------------------
    m = _max_selective(p)

    # --- assert ------------------------------------------
    assert np.allclose(m, np.array([0.0, 0.345, 0.0, 0.345, 0.0, 0.0], dtype=np.float32))
    assert m is not p, "Returned array should be a new array, not a reference to the input."


# =================================================================================================
#  Modify Selectivity
# =================================================================================================
@pytest.mark.parametrize(
    "modify_p_selectivity_fun",
    [
        modify_p_selectivity_power,
        modify_p_selectivity_pwl2,
    ],
)
@pytest.mark.parametrize(
    "p",
    [
        np.array([0.5, 0.3, 0.2], dtype=np.float32),
        np.array([0.0, 1.7, 1.7, 0.9, 0.3], dtype=np.float32),
    ],
)
@pytest.mark.parametrize("modify", [0.743, -0.234, 0.1])
def test_modify_p_selectivity_invertible(modify_p_selectivity_fun: Callable, p: np.ndarray, modify: float):
    # --- act ---------------------------------------------
    p_1 = modify_p_selectivity_fun(p, np.float32(modify))  # apply 'modify'
    p_2 = modify_p_selectivity_fun(p_1, -np.float32(modify))  # apply '-modify'

    # --- assert ------------------------------------------
    assert np.allclose(p, p_2)


@pytest.mark.parametrize(
    "modify_p_selectivity_fun",
    [
        modify_p_selectivity_power,
        modify_p_selectivity_pwl2,
    ],
)
@pytest.mark.parametrize(
    "modify_less, modify_more",
    [
        (-1.1, -0.9),
        (-1.0, -0.9),
        (-0.9, -0.5),
        (-0.5, -0.1),
        (-0.1, 0.0),
        (0.0, 0.1),
        (0.1, 0.5),
        (0.5, 0.9),
        (0.9, 1.0),
        (0.9, 1.1),
    ],
)
def test_modify_p_selectivity_relative(modify_p_selectivity_fun: Callable, modify_less: float, modify_more: float):
    # --- arrange -----------------------------------------
    p = np.array([0.0, 1.7, 1.7, 0.9, 0.3], dtype=np.float32)
    p_max = _p_max(p)

    # --- act ---------------------------------------------
    p_less = modify_p_selectivity_fun(p, np.float32(modify_less))
    p_more = modify_p_selectivity_fun(p, np.float32(modify_more))

    # --- assert ------------------------------------------
    assert max(p_less) == pytest.approx(p_max, rel=1e-5)  # higher tolerance due to float32 precision
    assert max(p_more) == pytest.approx(p_max, rel=1e-5)  # higher tolerance due to float32 precision
    assert sum(p_less) > sum(p_more)  # same maximum, but higher total sum --> less selective
    for i in range(p.size):
        if p[i] == p_max:
            assert p_less[i] == pytest.approx(p_max, rel=1e-5)  # higher tolerance due to float32 precision
            assert p_more[i] == pytest.approx(p_max, rel=1e-5)  # higher tolerance due to float32 precision


@pytest.mark.parametrize(
    "modify_p_selectivity_fun",
    [
        modify_p_selectivity_power,
        modify_p_selectivity_pwl2,
    ],
)
@pytest.mark.parametrize(
    "p, modify, p_expected",
    [
        (np.array([0.5, 0.3, 0.2], dtype=np.float32), -1.1, np.array([0.5, 0.5, 0.5], dtype=np.float32)),
        (np.array([0.5, 0.3, 0.2], dtype=np.float32), 0.0, np.array([0.5, 0.3, 0.2], dtype=np.float32)),
        (np.array([0.5, 0.3, 0.2], dtype=np.float32), 1.1, np.array([0.5, 0.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), -1.1, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), -1.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), -0.5, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.5, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), 1.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), 1.1, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        (np.array([-1.0, -2.0, -3.0], dtype=np.float32), 0.7, np.array([-1.0, -2.0, -3.0], dtype=np.float32)),
    ],
)
def test_modify_p_selectivity_edge_cases(
    modify_p_selectivity_fun: Callable, p: np.ndarray, modify, p_expected: np.ndarray
):
    # --- act ---------------------------------------------
    p_mod = modify_p_selectivity_fun(p, np.float32(modify))

    # --- assert ------------------------------------------
    assert np.allclose(p_mod, p_expected)
