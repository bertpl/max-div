from max_div.internal.utils import EPS, HALF_EPS


def test_eps():
    assert 1.0 + EPS != 1.0
    assert 1.0 + (EPS / 2) == 1.0


def test_half_eps():
    assert 0.9 * EPS < (HALF_EPS * HALF_EPS) < 1.1 * EPS
