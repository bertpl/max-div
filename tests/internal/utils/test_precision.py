import numpy as np

from max_div.internal.utils import ALMOST_ONE, ALMOST_ONE_F32, EPS, EPS_F32, HALF_EPS, HALF_EPS_F32


# -------------------------------------------------------------------------
#  64-bit floats
# -------------------------------------------------------------------------
def test_eps():
    assert isinstance(EPS, float)
    assert 1.0 + EPS != 1.0
    assert 1.0 + (EPS / 2) == 1.0


def test_half_eps():
    assert isinstance(HALF_EPS, float)
    assert 0.9 * EPS < (HALF_EPS * HALF_EPS) < 1.1 * EPS


def test_almost_one():
    assert isinstance(ALMOST_ONE, float)
    assert ALMOST_ONE < 1.0 < ALMOST_ONE + (10 * EPS)


# -------------------------------------------------------------------------
#  32-bit floats
# -------------------------------------------------------------------------
def test_eps_f32():
    assert isinstance(EPS_F32, np.float32)
    assert np.float32(1.0 + EPS_F32) != np.float32(1.0)
    assert np.float32(1.0 + (EPS_F32 / 2)) == np.float32(1.0)


def test_half_eps_f32():
    assert isinstance(HALF_EPS_F32, np.float32)
    assert np.float32(0.9 * EPS_F32) < np.float32(HALF_EPS_F32 * HALF_EPS_F32) < np.float32(1.1 * EPS_F32)


def test_almost_one_f32():
    assert isinstance(ALMOST_ONE_F32, np.float32)
    assert np.float32(ALMOST_ONE_F32) < np.float32(1.0) < np.float32(ALMOST_ONE_F32 + (10 * EPS_F32))
