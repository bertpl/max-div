"""The solver picks its distance backend here, and builds the store that choice implies.

AUTO semantics differ per problem flavor, deliberately:

  - Vector problems: distances are an internal artifact the user never sees, so AUTO transparently
    picks the fastest layout that fits in memory (full matrix, else condensed, else lazy).
  - Distance problems: the provided format may encode a conscious memory decision, so AUTO keeps
    it as given — condensed stays condensed, a square matrix stays a full matrix, zero-copy.

The memory criterion compares backend bytes against a fraction of *total* physical RAM: total is
cheap and stable to probe (unlike available memory, which fluctuates and is awkward to read on
some platforms), and the conservative fraction absorbs the machine load the probe deliberately
ignores.  Overshooting would page — worse than the foregone speedup — while undershooting only
concedes a narrow band of problem sizes that explicit pinning recovers.
"""

import os
from enum import StrEnum
from typing import ClassVar

from max_div._core.metrics._distance import (
    KIND_CONDENSED,
    KIND_FULL_MATRIX,
    DistanceStore,
    SharedDistanceStore,
    compute_full_matrix,
    compute_pdist,
    expand_condensed,
    publish_distance_store,
)
from max_div._core.problem import MaxDivProblem, VectorMaxDivProblem

# fraction of total physical RAM a stored backend may claim under AUTO
_AUTO_MEMORY_FRACTION = 1 / 3


# =================================================================================================
#  DistanceStorage
# =================================================================================================
class DistanceStorage(StrEnum):
    """How the solver stores pairwise distances during search.

    `AUTO` (the default) lets max-div decide: for vector problems, the fastest layout that fits in
    memory; for distance-input problems, the format the distances were provided in.  The resolved
    backend is reported in the solution summary.  Pinning a specific backend overrides the policy —
    `LAZY` requires vectors, so it is unavailable for distance-input problems.
    """

    AUTO = "auto"
    CONDENSED = "condensed"
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
        total_memory_bytes: total physical RAM, or None when unknown (degrades to condensed).
    """
    if storage != DistanceStorage.AUTO:
        return storage
    if not isinstance(problem, VectorMaxDivProblem):
        # distance-input problems: keep the format the user provided
        return DistanceStorage.FULL_MATRIX if problem.distance_store().matrix.size else DistanceStorage.CONDENSED
    if total_memory_bytes is None:
        return DistanceStorage.CONDENSED  # probe failed on this platform: the proven default
    budget = total_memory_bytes * _AUTO_MEMORY_FRACTION
    n = problem.n
    if 4 * n * n <= budget:
        return DistanceStorage.FULL_MATRIX
    if 2 * n * n <= budget:
        return DistanceStorage.CONDENSED
    return DistanceStorage.LAZY


def build_distance_store(problem: MaxDivProblem, resolved: DistanceStorage) -> DistanceStore:
    """Build the distance store for an already-resolved (non-AUTO) backend choice.

    Zero-copy wherever the problem already holds the data in the requested layout; conversions
    (e.g. condensed input with a full matrix requested) allocate consciously here.

    Raises:
        ValueError: For LAZY on a distance-input problem (no vectors to compute from), or when the
            requested stored backend cannot fit in physical memory at all.
    """
    is_vector_problem = isinstance(problem, VectorMaxDivProblem)
    n = problem.n
    match resolved:
        case DistanceStorage.CONDENSED:
            if is_vector_problem:
                _check_fits_physical_memory(resolved, _stored_backend_bytes(resolved, n), is_vector_problem)
            return DistanceStore.condensed(problem.condensed_distances(), n)
        case DistanceStorage.FULL_MATRIX:
            if is_vector_problem:
                _check_fits_physical_memory(resolved, _stored_backend_bytes(resolved, n), is_vector_problem)
                return DistanceStore.full_matrix_from_vectors(problem.vectors, problem.distance_metric)
            as_given = problem.distance_store()
            if as_given.matrix.size:
                return as_given  # square input: already a full matrix, zero-copy
            _check_fits_physical_memory(resolved, _stored_backend_bytes(resolved, n), is_vector_problem)
            return DistanceStore.full_matrix_from_condensed(as_given.pdist, n)
        case DistanceStorage.LAZY:
            if not is_vector_problem:
                raise ValueError(
                    "Lazy distance storage computes distances from vectors, which a distance-input "
                    "problem does not have; choose CONDENSED or FULL_MATRIX, or construct the "
                    "problem from vectors."
                )
            return DistanceStore.lazy(problem.vectors, problem.distance_metric)
        case _:
            raise ValueError(f"Distance storage must be resolved before building a store; got {resolved}.")


def build_shared_distance_store(problem: MaxDivProblem, resolved: DistanceStorage) -> SharedDistanceStore:
    """Build the store for an already-resolved backend in shared memory, for several processes to read.

    Computed backends are built straight into the segment: at full-matrix sizes a build-then-copy
    would double peak resident memory for its duration.  Data the problem already holds is copied in
    instead, since the bytes have to live in the segment.

    The caller owns the returned segment and must keep it open for as long as any process reads it.

    Raises:
        ValueError: as `build_distance_store`, for a backend the problem cannot provide.
    """
    n = problem.n
    if isinstance(problem, VectorMaxDivProblem):
        match resolved:
            case DistanceStorage.CONDENSED:
                _check_fits_physical_memory(resolved, _stored_backend_bytes(resolved, n), True)
                shared = SharedDistanceStore.allocate(((n * (n - 1)) // 2,), KIND_CONDENSED, n)
                compute_pdist(problem.vectors, problem.distance_metric, out=shared.buffer)
                return shared
            case DistanceStorage.FULL_MATRIX:
                _check_fits_physical_memory(resolved, _stored_backend_bytes(resolved, n), True)
                shared = SharedDistanceStore.allocate((n, n), KIND_FULL_MATRIX, n)
                compute_full_matrix(problem.vectors, problem.distance_metric, out=shared.buffer)
                return shared
            case DistanceStorage.LAZY:
                # `lazy` applies the per-metric preparation, so its output is what gets published:
                # the segment must hold the vectors in the form the distance reads expect.
                return publish_distance_store(DistanceStore.lazy(problem.vectors, problem.distance_metric), n)
    elif resolved == DistanceStorage.FULL_MATRIX:
        as_given = problem.distance_store()
        if not as_given.matrix.size:
            # condensed input, full matrix asked for: expanding into the segment makes it the one
            # n²-sized allocation on this path
            _check_fits_physical_memory(resolved, _stored_backend_bytes(resolved, n), False)
            shared = SharedDistanceStore.allocate((n, n), KIND_FULL_MATRIX, n)
            expand_condensed(as_given.pdist, n, out=shared.buffer)
            return shared
    return publish_distance_store(build_distance_store(problem, resolved), n)


def _stored_backend_bytes(resolved: DistanceStorage, n: int) -> int:
    """Return the bytes a stored backend claims for n items."""
    return 4 * n * n if resolved == DistanceStorage.FULL_MATRIX else 2 * n * n


def _check_fits_physical_memory(resolved: DistanceStorage, bytes_needed: int, is_vector_problem: bool) -> None:
    """Raise early, with the remedy named, when a stored backend cannot fit in physical RAM at all.

    Only guards allocations this module is about to make; adopted user arrays already exist.
    """
    total = total_physical_memory_bytes()
    if total is not None and bytes_needed > total:
        lazy_hint = " or DistanceStorage.LAZY (no O(n²) memory)" if is_vector_problem else ""
        raise ValueError(
            f"Distance storage '{resolved.value}' needs ~{bytes_needed / 2**30:.1f} GiB, but this machine "
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
