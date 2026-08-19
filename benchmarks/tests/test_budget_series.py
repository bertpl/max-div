from itertools import pairwise

import pytest

from benchmarks.common import iteration_budget_series, time_budget_series


def test_time_budget_series_covers_ceiling():
    """The series starts at t_min and its last value covers the ceiling."""
    # --- act ------------------------------
    budgets = time_budget_series(0.001, 1.0, factor=2.0)

    # --- assert ---------------------------
    assert budgets[0] == 0.001
    assert budgets[-1] >= 1.0
    assert all(b == pytest.approx(2 * a) for a, b in pairwise(budgets))


def test_iteration_budget_series_strictly_increases():
    """Iteration budgets grow geometrically and always advance by at least one."""
    # --- act ------------------------------
    budgets = iteration_budget_series(1, 1000, factor=1.3)

    # --- assert ---------------------------
    assert budgets[0] == 1
    assert budgets[-1] >= 1000
    assert all(b > a for a, b in pairwise(budgets))


@pytest.mark.parametrize("kwargs", [{"t_min_sec": 0.0}, {"t_max_sec": 0.0001}, {"factor": 1.0}])
def test_time_budget_series_rejects_bad_arguments(kwargs):
    """The series starts at t_min and its last value covers the ceiling."""
    # --- act / assert ---------------------
    with pytest.raises(ValueError, match="A budget series requires"):
        time_budget_series(**{"t_min_sec": 0.001, "t_max_sec": 1.0, "factor": 2.0, **kwargs})


@pytest.mark.parametrize("kwargs", [{"i_min": 0}, {"i_max": 5}, {"factor": 0.9}])
def test_iteration_budget_series_rejects_bad_arguments(kwargs):
    """Iteration budgets grow geometrically and always advance by at least one."""
    # --- act / assert ---------------------
    with pytest.raises(ValueError, match="A budget series requires"):
        iteration_budget_series(**{"i_min": 10, "i_max": 1000, "factor": 2.0, **kwargs})
