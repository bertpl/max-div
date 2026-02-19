import numpy as np

from max_div._core._math.modify_p_selectivity._helpers import _p_max


def test_modify_p_helpers_p_max():
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
