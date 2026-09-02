import numpy as np
import pytest

from max_div._core._math.modify_p_selectivity._exponential import exponential_selectivity


@pytest.mark.parametrize("low_value", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("modifier", [-0.8, -0.4, 0.0, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("descending", [False, True])
def test_order_based_selectivity(low_value: float, modifier: float, descending: bool):
    # --- arrange ----------------------
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

    # --- act --------------------------
    p_out = np.empty_like(p_in)
    exponential_selectivity(p_in, p_out, np.float32(modifier), descending, np.float32(low_value))

    # --- assert -----------------------
    assert np.allclose(p_out, expected_p_out, rtol=0.01, atol=0.0)  # fast_exp2_f32 has relative error <1%


def test_order_based_selectivity_corner_case():
    # --- arrange ----------------------
    p_in = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    expected_p_out = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # --- act --------------------------
    p_out = np.empty_like(p_in)
    exponential_selectivity(p_in, p_out, np.float32(0.5), reverse=False, low_value=np.float32(0.1))

    # --- assert -----------------------
    assert np.array_equal(p_out, expected_p_out)


@pytest.mark.parametrize(
    "p_in",
    [
        np.array([np.inf], dtype=np.float32),
        np.array([np.inf, 3.0], dtype=np.float32),
        np.array([np.inf, np.inf], dtype=np.float32),
        np.array([np.nan, 1.0], dtype=np.float32),
    ],
)
def test_non_finite_inputs_fall_back_to_uniform(p_in: np.ndarray):
    # a sole selected item has +inf diversity contribution; the transform must
    # degrade to uniform probabilities instead of emitting NaNs (solver crash)
    # --- arrange ----------------------
    expected_p_out = np.ones_like(p_in)

    # --- act --------------------------
    p_out = np.empty_like(p_in)
    exponential_selectivity(p_in, p_out, np.float32(0.5), reverse=True, low_value=np.float32(0.1))

    # --- assert -----------------------
    assert np.array_equal(p_out, expected_p_out)


def test_near_maximal_selectivity_cuts_off_the_low_end():
    """Near modifier = 1 the exponential underflows float32 to 0.0 for the low end of the range."""
    # --- arrange ----------------------
    p_in = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    p_out = np.empty_like(p_in)

    # --- act --------------------------
    exponential_selectivity(p_in=p_in, p_out=p_out, modifier=np.float32(0.99), reverse=False, low_value=np.float32(0.1))

    # --- assert -----------------------
    assert p_out[-1] > 0.99  # p_in = 1.0 maps to 1.0 up to the fast exponential's error
    assert (p_out[p_in <= 0.7] == 0.0).all()  # the low end is cut off entirely
    assert (p_out[p_in >= 0.85] > 0.0).all()  # the top of the range stays above zero
