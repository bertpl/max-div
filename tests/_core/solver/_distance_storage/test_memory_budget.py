import pytest

from max_div._core.solver._distance_storage.memory_budget import (
    check_fits_physical_memory,
    full_matrix_bytes,
    total_physical_memory_bytes,
)

GIB = 2**30


def test_full_matrix_bytes_is_four_per_pair_including_the_diagonal():
    """Each (i, j) cell costs four bytes, the diagonal included."""
    # --- act / assert -----------------
    assert full_matrix_bytes(1_000) == 4_000_000


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
        check_fits_physical_memory(full_matrix_bytes(2_000_000), lazy_available)  # ~16 TiB
    assert ("LAZY" in str(excinfo.value)) is lazy_available


def test_check_fits_physical_memory_accepts_a_small_matrix():
    """A matrix that fits passes silently."""
    # --- act / assert -----------------
    check_fits_physical_memory(full_matrix_bytes(10), lazy_available=True)  # no raise
