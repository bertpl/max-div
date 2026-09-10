"""The solver picks its distance backend here, and builds the store that choice implies.

AUTO semantics differ per problem flavor, deliberately:

  - Vector problems: distances are an internal artifact the user never sees, so AUTO transparently
    picks the full matrix when it fits in memory and computes distances on demand otherwise.
  - Distance problems: the distances exist already, so AUTO stores them as a full matrix — a square
    input zero-copy, a condensed input expanded.

The memory criterion compares the matrix bytes against a fraction of *total* physical RAM: total is
cheap and stable to probe (unlike available memory, which fluctuates and is awkward to read on
some platforms), and the conservative fraction absorbs the machine load the probe deliberately
ignores.  Overshooting would page — worse than the foregone speedup — while undershooting only
concedes a narrow band of problem sizes that explicit pinning recovers.
"""

import os
from enum import StrEnum
from typing import ClassVar

from max_div._core.metrics._distance import (
    KIND_FULL_MATRIX,
    DistanceStore,
    SharedDistanceStore,
    publish_distance_store,
)
from max_div._core.problem import MaxDivProblem, VectorMaxDivProblem

# fraction of total physical RAM the full matrix may claim under AUTO
_AUTO_MEMORY_FRACTION = 1 / 3


# =================================================================================================
#  DistanceStorage
# =================================================================================================
class DistanceStorage(StrEnum):
    """How the solver stores pairwise distances during search.

    `AUTO` (the default) lets max-div decide: for vector problems, the full matrix when it fits in
    memory and the lazy backend otherwise; for distance-input problems, always the full matrix,
    which expands a condensed input at twice its memory.  The resolved backend is reported in the
    solution summary.  Pinning a specific backend overrides the policy — `LAZY` requires vectors,
    so it is unavailable for distance-input problems.
    """

    AUTO = "auto"
    FULL_MATRIX = "full_matrix"
    LAZY = "lazy"


# =================================================================================================
#  Resolution & construction
# =================================================================================================
def select_distance_storage(
    problem: MaxDivProblem, storage: DistanceStorage, total_memory_bytes: int | None
) -> DistanceStorage:
    """Select a concrete backend for the given problem when the choice is `AUTO`; explicit choices pass through.

    A pure function of its arguments — the memory probe is injected, so the selection is
    deterministic and testable.

    Args:
        problem: the problem to be solved.
        storage: the user's choice, possibly AUTO.
        total_memory_bytes: total physical RAM, or None when unknown (degrades to lazy, the one
            backend that cannot page).
    """
    if storage != DistanceStorage.AUTO:
        return storage
    if not isinstance(problem, VectorMaxDivProblem):
        return DistanceStorage.FULL_MATRIX  # distance-input problems have no vectors, so this is their only backend
    if total_memory_bytes is None:
        return DistanceStorage.LAZY
    if full_matrix_bytes(problem.n) <= total_memory_bytes * _AUTO_MEMORY_FRACTION:
        return DistanceStorage.FULL_MATRIX
    return DistanceStorage.LAZY


def build_distance_store(problem: MaxDivProblem, resolved: DistanceStorage) -> DistanceStore:
    """Build the distance store for an already-resolved (non-AUTO) backend choice.

    The full matrix is computed for a vector problem, adopted zero-copy for a square input, and
    expanded for a condensed input; the memory check guards the two allocating cases.

    Raises:
        ValueError: For LAZY on a distance-input problem (no vectors to compute from), or when the
            full matrix cannot fit in physical memory at all.
    """
    match resolved:
        case DistanceStorage.FULL_MATRIX:
            if not problem.has_full_matrix:
                _check_fits_physical_memory(problem)
            return DistanceStore.full_matrix(problem.full_matrix())
        case DistanceStorage.LAZY:
            if not isinstance(problem, VectorMaxDivProblem):
                raise ValueError(
                    "Lazy distance storage computes distances from vectors, which a distance-input "
                    "problem does not have; choose FULL_MATRIX, or construct the problem from vectors."
                )
            return DistanceStore.lazy_from_vectors(problem.vectors, problem.distance_metric)
        case _:
            raise ValueError(f"Distance storage must be resolved before building a store; got {resolved}.")


def build_shared_distance_store(problem: MaxDivProblem, resolved: DistanceStorage) -> SharedDistanceStore:
    """Build the store for an already-resolved backend in shared memory, for several processes to read.

    A full matrix is built or expanded straight into the segment: at full-matrix sizes a
    build-then-copy would double peak resident memory for its duration.  Data the problem already
    holds in final form is copied in, since the bytes have to live in the segment.

    The caller owns the returned segment and must keep it open for as long as any process reads it.

    Raises:
        ValueError: as `build_distance_store`, for a backend the problem cannot provide.
    """
    if resolved == DistanceStorage.FULL_MATRIX and not problem.has_full_matrix:
        _check_fits_physical_memory(problem)
        shared = SharedDistanceStore.allocate((problem.n, problem.n), KIND_FULL_MATRIX)
        problem.full_matrix(out=shared.buffer)
        return shared
    # The remaining cases, a full matrix as given or lazy, publish a built store; a lazy store holds
    # the vectors preprocessed for the metric, so its array is the one published — the segment
    # must hold what the distance reads expect.
    return publish_distance_store(build_distance_store(problem, resolved))


def full_matrix_bytes(n: int) -> int:
    """Return the bytes a full float32 distance matrix claims for n items."""
    return 4 * n * n


def _check_fits_physical_memory(problem: MaxDivProblem) -> None:
    """Raise early, with the remedy named, when the full matrix cannot fit in physical RAM at all.

    Only guards allocations this module is about to make; adopted user arrays already exist.
    """
    total = total_physical_memory_bytes()
    bytes_needed = full_matrix_bytes(problem.n)
    if total is not None and bytes_needed > total:
        lazy_hint = " or DistanceStorage.LAZY (no O(n²) memory)" if isinstance(problem, VectorMaxDivProblem) else ""
        raise ValueError(
            f"Distance storage 'full_matrix' needs ~{bytes_needed / 2**30:.1f} GiB, but this machine "
            f"has {total / 2**30:.1f} GiB of physical memory; choose a smaller problem{lazy_hint}."
        )


def total_physical_memory_bytes() -> int | None:
    """Return total physical RAM in bytes via the stdlib, or None when it cannot be determined.

    POSIX exposes it through sysconf; Windows through one kernel32 call.  Callers must treat None
    as "unknown" and degrade gracefully.
    """
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
