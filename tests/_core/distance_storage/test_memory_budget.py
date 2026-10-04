import pytest

from max_div._core.distance_storage.memory_budget import (
    check_fits_physical_memory,
    data_matrix_bytes,
    full_matrix_bytes,
    total_physical_memory_bytes,
)

GIB = 2**30


def test_full_matrix_bytes_is_four_per_pair_including_the_diagonal():
    """Each (i, j) cell costs four bytes, the diagonal included."""
    # --- act / assert -----------------
    assert full_matrix_bytes(1_000) == 4_000_000


@pytest.mark.parametrize(
    "shape, expected", [((1_000, 3), 12_000), ((1_000, 0), 0), ((2_000_000, 2_000_000), 16 * 10**12)]
)
def test_data_matrix_bytes_is_four_per_element(shape: tuple[int, ...], expected: int):
    """A float32 data matrix costs four bytes per element, without overflow for a matrix larger than RAM."""
    # --- act / assert -----------------
    assert data_matrix_bytes(shape) == expected


def test_total_physical_memory_bytes_on_this_platform():
    """On every CI platform the stdlib probe must return at least 1 GiB."""

    # --- act --------------------------
    total = total_physical_memory_bytes()

    # --- assert -----------------------
    assert total is not None
    assert total >= 1 * GIB


@pytest.mark.parametrize("lazy_available", [True, False], ids=["vectors", "distances"])
def test_check_fits_physical_memory_rejects_a_matrix_larger_than_ram_and_names_the_remedy(lazy_available: bool):
    """A matrix larger than all physical RAM is refused early; the lazy remedy is named only when it exists."""

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="physical memory") as excinfo:
        check_fits_physical_memory(full_matrix_bytes(2_000_000), 64 * GIB, lazy_available)  # ~16 TiB
    assert ("LAZY" in str(excinfo.value)) is lazy_available


@pytest.mark.parametrize("total_memory_bytes", [64 * GIB, None], ids=["fits", "unknown-ram"])
def test_check_fits_physical_memory_accepts_a_matrix_that_fits_or_unknown_ram(total_memory_bytes: int | None):
    """A matrix that fits passes silently, and so does any matrix when the total RAM is unknown."""
    # --- act / assert -----------------
    check_fits_physical_memory(full_matrix_bytes(10), total_memory_bytes, lazy_available=True)  # no raise
