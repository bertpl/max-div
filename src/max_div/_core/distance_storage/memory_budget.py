"""This module sizes a full matrix, probes total physical RAM, and refuses a matrix that cannot fit."""

import os
from typing import ClassVar

# Under AUTO the full matrices together may claim this fraction of *total* physical RAM.  Total is cheap and
# stable to probe, unlike available memory, which fluctuates and is awkward to read on some
# platforms; the conservative fraction absorbs the machine load the probe deliberately ignores.
# Overshooting would page — worse than the foregone speedup — while undershooting only loses the
# full matrix for problems just above the threshold, where the user can pin `FULL_MATRIX` explicitly.
AUTO_MEMORY_FRACTION = 1 / 3


def full_matrix_bytes(n: int) -> int:
    """Return the bytes a full float32 distance matrix claims for n items."""
    return 4 * n * n


def check_fits_physical_memory(bytes_needed: int, lazy_available: bool) -> None:
    """Raise early, with the remedy named, when the full matrices cannot fit in physical RAM at all.

    Args:
        bytes_needed: the bytes the allocations will claim together.
        lazy_available: whether the problem has vectors, so the lazy storage type can be named as the remedy.
    """
    total = total_physical_memory_bytes()
    if total is not None and bytes_needed > total:
        lazy_hint = " or DistanceStorageType.LAZY (no O(n²) memory)" if lazy_available else ""
        raise ValueError(
            f"Distance storage 'full_matrix' needs ~{bytes_needed / 2**30:.1f} GiB, but this machine "
            f"has {total / 2**30:.1f} GiB of physical memory; choose a smaller problem{lazy_hint}."
        )


def total_physical_memory_bytes() -> int | None:
    """Return total physical RAM in bytes via the stdlib, or None when it cannot be determined."""
    # --- POSIX ----------------------------------
    try:
        page_count = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if page_count > 0 and page_size > 0:
            return page_count * page_size
    except (ValueError, OSError, AttributeError):
        pass
    # --- Windows --------------------------------
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            # ClassVar: the ctypes protocol reads _fields_ from the class, never per-instance
            _fields_: ClassVar = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # ty: ignore[unresolved-attribute]
            return int(status.ullTotalPhys)
    except (AttributeError, OSError):
        pass  # ctypes.windll only exists on Windows
    return None
