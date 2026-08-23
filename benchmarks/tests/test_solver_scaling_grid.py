import pytest

from benchmarks.solver_scaling.grid import GRID_MIN, MEMORY_CAP_BYTES, operational_bound, size_grid


def test_size_grid_is_the_1_2_5_sequence_floored_at_the_minimum():
    # --- act --------------------------
    grid = size_grid(2000)

    # --- assert -----------------------
    assert grid == [20, 50, 100, 200, 500, 1000, 2000]
    assert grid[0] == GRID_MIN


def test_size_grid_includes_the_bound_only_when_it_lies_on_the_grid():
    # --- act / assert -----------------
    assert size_grid(300) == [20, 50, 100, 200]
    assert size_grid(500)[-1] == 500


def test_size_grid_rejects_a_bound_below_the_floor():
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="below the smallest grid size"):
        size_grid(GRID_MIN - 1)


def test_operational_bound_is_where_the_raw_vectors_fill_the_cap():
    # --- act / assert -----------------
    assert operational_bound(d=2) == MEMORY_CAP_BYTES // 8
    assert operational_bound(d=4) == MEMORY_CAP_BYTES // 16
