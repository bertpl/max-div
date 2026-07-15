from itertools import pairwise

import pytest

from benchmarks.common import iteration_ladder, time_ladder


def test_time_ladder_covers_ceiling():
    # --- act ---------------------------------------------
    rungs = time_ladder(0.001, 1.0, factor=2.0)

    # --- assert ------------------------------------------
    assert rungs[0] == 0.001
    assert rungs[-1] >= 1.0
    assert all(b == pytest.approx(2 * a) for a, b in pairwise(rungs))


def test_iteration_ladder_strictly_increases():
    # --- act ---------------------------------------------
    rungs = iteration_ladder(1, 1000, factor=1.3)

    # --- assert ------------------------------------------
    assert rungs[0] == 1
    assert rungs[-1] >= 1000
    assert all(b > a for a, b in pairwise(rungs))


@pytest.mark.parametrize("kwargs", [{"t_min_sec": 0.0}, {"t_max_sec": 0.0001}, {"factor": 1.0}])
def test_time_ladder_rejects_bad_arguments(kwargs):
    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="Ladder requires"):
        time_ladder(**{"t_min_sec": 0.001, "t_max_sec": 1.0, "factor": 2.0, **kwargs})


@pytest.mark.parametrize("kwargs", [{"i_min": 0}, {"i_max": 5}, {"factor": 0.9}])
def test_iteration_ladder_rejects_bad_arguments(kwargs):
    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="Ladder requires"):
        iteration_ladder(**{"i_min": 10, "i_max": 1000, "factor": 2.0, **kwargs})
