"""Guards for the ceilings campaign's candidate-size grid and its constants."""

import pytest

from benchmarks.ceilings.grid import GRID_FLOOR, MEMORY_CAP_BYTES, operational_bound, size_grid


def test_grid_is_three_values_per_decade_from_the_floor() -> None:
    # --- act --------------------------
    sizes = size_grid(5000)

    # --- assert -----------------------
    assert sizes == [100, 200, 500, 1000, 2000, 5000]


def test_grid_respects_a_bound_between_values() -> None:
    """The bound caps the grid; it does not have to be a grid value itself."""
    # --- act / assert -----------------
    assert size_grid(4999)[-1] == 2000
    assert size_grid(100) == [100]


def test_a_bound_below_the_floor_is_rejected() -> None:
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="grid floor"):
        size_grid(GRID_FLOOR - 1)


def test_operational_bound_is_where_the_raw_vectors_fill_the_cap() -> None:
    """The campaign never generates an instance whose float32 input alone busts 32 GB."""
    # --- act --------------------------
    bound = operational_bound(d=2)

    # --- assert -----------------------
    assert bound * 2 * 4 <= MEMORY_CAP_BYTES < (bound + 1) * 2 * 4
