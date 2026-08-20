"""Guards for the tool-scaling benchmarks' candidate-size grid and its constants."""

import pytest

from benchmarks.tool_scaling.grid import GRID_MIN, MEMORY_CAP_BYTES, operational_bound, size_grid


def test_grid_is_three_values_per_decade_from_the_minimum() -> None:
    """The grid runs 1-2-5 per decade, starting at 100."""
    # --- act --------------------------
    sizes = size_grid(5000)

    # --- assert -----------------------
    assert sizes == [100, 200, 500, 1000, 2000, 5000]


def test_grid_respects_a_bound_between_values() -> None:
    """The bound caps the grid; it does not have to be a grid value itself."""
    # --- act / assert -----------------
    assert size_grid(4999)[-1] == 2000
    assert size_grid(100) == [100]


def test_a_bound_below_the_minimum_is_rejected() -> None:
    """A bound under the smallest grid size is a caller error, not an empty grid."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="smallest grid size"):
        size_grid(GRID_MIN - 1)


def test_operational_bound_is_where_the_raw_vectors_fill_the_cap() -> None:
    """The campaign never generates an instance whose float32 input alone busts 32 GB."""
    # --- act --------------------------
    bound = operational_bound(d=2)

    # --- assert -----------------------
    assert bound * 2 * 4 <= MEMORY_CAP_BYTES < (bound + 1) * 2 * 4
