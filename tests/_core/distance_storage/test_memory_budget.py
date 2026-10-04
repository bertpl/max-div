import pytest

from max_div._core.distance_storage.memory_budget import (
    data_matrix_bytes,
    total_physical_memory_bytes,
)

GIB = 2**30


@pytest.mark.parametrize(
    "shape, expected", [((1_000, 3), 12_000), ((1_000, 0), 0), ((2_000_000, 2_000_000), 16 * 10**12)]
)
def test_data_matrix_bytes_is_four_per_element(shape: tuple[int, ...], expected: int):
    """A float32 data matrix costs 4 bytes per element, without overflow for a matrix larger than RAM."""
    # --- act / assert -----------------
    assert data_matrix_bytes(shape) == expected


def test_total_physical_memory_bytes_on_this_platform():
    """On every CI platform the stdlib probe must return at least 1 GiB."""

    # --- act --------------------------
    total = total_physical_memory_bytes()

    # --- assert -----------------------
    assert total is not None
    assert total >= 1 * GIB
