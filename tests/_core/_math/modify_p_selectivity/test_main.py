import numpy as np
import pytest

from max_div._core._math.modify_p_selectivity._main import modify_p_selectivity

METHODS_AND_TOLERANCES = [
    (np.int32(0), 1e-6),
    (np.int32(10), 1e-2),
    (np.int32(20), 1e-2),
    (np.int32(100), 1e-1),
]


@pytest.mark.parametrize("method, tol", METHODS_AND_TOLERANCES)
def test_modify_p_selectivity_accuracy(method, tol):
    # --- arrange -----------------------------------------
    n = 50
    p = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    modifiers = np.linspace(-0.9, 0.9, num=n, dtype=np.float32)

    p_out_expected = [np.array([p[i] ** ((1 + m) / (1 - m)) for i in range(n)], dtype=np.float32) for m in modifiers]

    # --- act ---------------------------------------------
    e_tot = 0.0
    for modifier, expected_result in zip(modifiers, p_out_expected):
        p_out = np.empty_like(p)
        modify_p_selectivity(p, modifier, method, p_out)
        e_tot += sum(abs(p_out - expected_result))

    e_tot /= n * n

    # --- assert ------------------------------------------
    assert e_tot <= tol


@pytest.mark.parametrize("method, tol", METHODS_AND_TOLERANCES)
@pytest.mark.parametrize("modifier", np.linspace(-0.9, 0.9, 20))
def test_modify_p_selectivity_preserve_order(method, tol, modifier):
    # --- arrange -----------------------------------------
    p = np.linspace(0.0, 1.0, num=10_000, dtype=np.float32)

    # --- act ---------------------------------------------
    p_out = np.empty_like(p)
    modify_p_selectivity(p=p, modifier=np.float32(modifier), method=method, p_out=p_out)

    # --- assert ------------------------------------------
    assert np.array_equal(p_out, sorted(p_out))


@pytest.mark.parametrize(
    "p, modifier, p_expected",
    [
        (np.array([0.5, 0.3, 0.2], dtype=np.float32), -1.1, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
        (np.array([0.5, 0.3, 0.2], dtype=np.float32), 0.0, np.array([1.0, 0.6, 0.4], dtype=np.float32)),
        (np.array([0.5, 0.3, 0.2], dtype=np.float32), 1.1, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
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
@pytest.mark.parametrize("method", [0, 10, 20, 100])
def test_modify_p_selectivity_edge_cases(p: np.ndarray, modifier, p_expected: np.ndarray, method: int):
    # --- act ---------------------------------------------
    p_out = np.empty_like(p)
    modify_p_selectivity(p=p, modifier=np.float32(modifier), method=np.int32(method), p_out=p_out)

    # --- assert ------------------------------------------
    assert np.allclose(p_out, p_expected)


@pytest.mark.parametrize("method, tol", METHODS_AND_TOLERANCES)
@pytest.mark.parametrize("modifier", [-0.999, 0.999])
def test_modify_p_selectivity_ill_conditioning(method: int, tol: float, modifier):
    # --- arrange -----------------------------------------
    p = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32)

    # --- act ---------------------------------------------
    p_out = np.empty_like(p)
    modify_p_selectivity(p, np.float32(modifier), np.int32(method), p_out)

    # --- assert ------------------------------------------
    if modifier < -0.9:
        assert 0.99 <= p_out[1] <= 1.0
        assert 0.99 <= p_out[2] <= 1.0
        assert 0.99 <= p_out[3] <= 1.0
    elif modifier > 0.9:
        assert 0.0 <= p_out[1] <= 0.01
        assert 0.0 <= p_out[2] <= 0.01
        assert 0.0 <= p_out[3] <= 0.01

    assert p_out[0] <= p_out[1] <= p_out[2] <= p_out[3] <= p_out[4]


def test_modify_p_selectivity_invalid_method():
    # --- arrange -----------------------------------------
    p = np.linspace(0.0, 1.0, num=10, dtype=np.float32)
    modifier = np.float32(0.5)
    method = np.int32(999)  # invalid method

    # --- act / assert ------------------------------------
    p_out = np.empty_like(p)
    with pytest.raises(NotImplementedError):
        modify_p_selectivity(p, modifier, method, p_out)
