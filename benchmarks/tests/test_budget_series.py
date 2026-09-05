from itertools import pairwise

import pytest

from benchmarks.common import grid_budget_series, iteration_budget_series, time_budget_series


def test_time_budget_series_covers_t_max():
    """The series starts at t_min, ends at or above t_max, and multiplies by the factor each step."""
    # --- act --------------------------
    budgets = time_budget_series(0.001, 1.0, factor=2.0)

    # --- assert -----------------------
    assert budgets[0] == 0.001
    assert budgets[-1] >= 1.0
    assert all(b == pytest.approx(2 * a) for a, b in pairwise(budgets))


def test_iteration_budget_series_strictly_increases():
    """A factor whose rounded step can be zero still yields strictly increasing counts."""
    # --- act --------------------------
    budgets = iteration_budget_series(1, 1000, factor=1.3)

    # --- assert -----------------------
    assert budgets[0] == 1
    assert budgets[-1] >= 1000
    assert all(b > a for a, b in pairwise(budgets))


@pytest.mark.parametrize("kwargs", [{"t_min_sec": 0.0}, {"t_max_sec": 0.0001}, {"factor": 1.0}])
def test_time_budget_series_rejects_bad_arguments(kwargs):
    """A non-positive start, a maximum below the start, or a factor <= 1 is rejected."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="A budget series requires"):
        time_budget_series(**{"t_min_sec": 0.001, "t_max_sec": 1.0, "factor": 2.0, **kwargs})


@pytest.mark.parametrize("kwargs", [{"i_min": 0}, {"i_max": 5}, {"factor": 0.9}])
def test_iteration_budget_series_rejects_bad_arguments(kwargs):
    """A non-positive start, a maximum below the start, or a factor <= 1 is rejected."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="A budget series requires"):
        iteration_budget_series(**{"i_min": 10, "i_max": 1000, "factor": 2.0, **kwargs})


def test_grid_budget_series_walks_1_2_5_and_ends_at_t_max():
    """The 1-2-5 series from 1 ms ends at exactly 60 s, replacing the 50 s grid point."""
    # --- act --------------------------
    budgets = grid_budget_series(0.001, 60.0)

    # --- assert -----------------------
    assert budgets[:4] == [0.001, 0.002, 0.005, 0.01]
    assert budgets[-3:] == [10.0, 20.0, 60.0]
    assert len(budgets) == 15


def test_grid_budget_series_from_one_second_has_six_points():
    """The multi-worker series shares the single-worker series' top six budgets."""
    # --- act / assert -----------------
    assert grid_budget_series(1.0, 60.0) == [1.0, 2.0, 5.0, 10.0, 20.0, 60.0]


@pytest.mark.parametrize("t_min, t_max", [(0.003, 1.0), (1.0, 1.0), (5.0, 2.0)])
def test_grid_budget_series_rejects_off_grid_or_inverted_bounds(t_min: float, t_max: float):
    """A start off the 1-2-5 grid, or an end not above the start, is rejected."""
    # --- act / assert -----------------
    with pytest.raises(ValueError):
        grid_budget_series(t_min, t_max)
