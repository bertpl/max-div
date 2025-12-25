import numpy as np
import pytest

from max_div.internal.math.powers_of_2 import _POWERS_OF_2_F32_MAX_EXP, _POWERS_OF_2_F32_MIN_EXP, power_of_2_f32


@pytest.mark.parametrize(
    "k",
    list(range(-149, 128)),  # 2**k with k in [-149, 127] can be represented as float32 >0 and <+inf
)
def test_power_of_2_f32_regular_range(k: int):
    # --- arrange -----------------------------------------
    expected_result = np.float32(2**k)

    # --- act ---------------------------------------------
    result = power_of_2_f32(np.int32(k))

    # --- assert ------------------------------------------
    assert result == expected_result


def test_power_of_2_f32_edge_cases():
    # --- act & assert ------------------------------------

    # underflow
    assert power_of_2_f32(np.int32(-1000000)) == 0.0
    assert power_of_2_f32(np.int32(-1000)) == 0.0
    assert power_of_2_f32(np.int32(-200)) == 0.0
    assert power_of_2_f32(np.int32(-151)) == 0.0
    assert power_of_2_f32(np.int32(-150)) == 0.0
    assert power_of_2_f32(_POWERS_OF_2_F32_MIN_EXP) == 0.0
    assert power_of_2_f32(np.int32(-149)) > 0.0

    # overflow
    assert power_of_2_f32(np.int32(127)) < np.inf
    assert power_of_2_f32(_POWERS_OF_2_F32_MAX_EXP) == np.inf
    assert power_of_2_f32(np.int32(128)) == np.inf
    assert power_of_2_f32(np.int32(129)) == np.inf
    assert power_of_2_f32(np.int32(150)) == np.inf
    assert power_of_2_f32(np.int32(200)) == np.inf
    assert power_of_2_f32(np.int32(1000)) == np.inf
    assert power_of_2_f32(np.int32(1000000)) == np.inf
