import numpy as np
import pytest

from max_div.internal.math.modify_p_selectivity._exponential import exponential_selectivity


@pytest.mark.parametrize("low_value", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("modifier", [-0.8, -0.4, 0.0, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("descending", [False, True])
def test_order_based_selectivity(low_value: float, modifier: float, descending: bool):
    # --- arrange -----------------------------------------
    p_in = np.array([1.0, 1.6, 2.0], dtype=np.float32)
    t = (1 + modifier) / (1 - modifier)

    if descending:
        expected_p_out = np.array(
            [low_value ** (t * 0.0), low_value ** (t * 0.6), low_value ** (t * 1.0)], dtype=np.float32
        )
    else:
        expected_p_out = np.array(
            [low_value ** (t * 1.0), low_value ** (t * 0.4), low_value ** (t * 0.0)], dtype=np.float32
        )

    # --- act ---------------------------------------------
    p_out = np.empty_like(p_in)
    exponential_selectivity(p_in, p_out, np.float32(modifier), descending, np.float32(low_value))

    # --- assert ------------------------------------------
    assert np.allclose(p_out, expected_p_out, rtol=0.01, atol=0.0)  # fast_exp2_f32 has relative error <1%


def test_order_based_selectivity_corner_case():
    # --- arrange -----------------------------------------
    p_in = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    expected_p_out = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # --- act ---------------------------------------------
    p_out = np.empty_like(p_in)
    exponential_selectivity(p_in, p_out, np.float32(0.5), descending=False, low_value=np.float32(0.1))

    # --- assert ------------------------------------------
    assert np.array_equal(p_out, expected_p_out)
