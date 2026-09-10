"""Resolve the user's storage choice, then build the store in process or into shared memory.

AUTO semantics differ per problem flavor, deliberately:

  - Vector problems: distances are an internal artifact the user never sees, so AUTO
    picks the full matrix when it fits in memory and computes distances on demand otherwise.
  - Distance problems: the distances exist already, so AUTO stores them as a full matrix — a square
    input zero-copy, a condensed input expanded.
"""

from max_div._core.metrics._distance import KIND_FULL_MATRIX, DistanceStore
from max_div._core.problem import MaxDivProblem, VectorMaxDivProblem

from .memory_budget import AUTO_MEMORY_FRACTION, check_fits_physical_memory, full_matrix_bytes
from .shared_memory import SharedDistanceStore, publish_distance_store
from .storage import DistanceStorage


# =================================================================================================
#  Resolution & construction
# =================================================================================================
def select_distance_storage(
    problem: MaxDivProblem, storage: DistanceStorage, total_memory_bytes: int | None
) -> DistanceStorage:
    """Select a concrete backend for the given problem when the choice is `AUTO`; explicit choices pass through.

    The memory probe is injected, so the selection is a pure function of its arguments and
    testable without the machine's RAM.

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
    if full_matrix_bytes(problem.n) <= total_memory_bytes * AUTO_MEMORY_FRACTION:
        return DistanceStorage.FULL_MATRIX
    return DistanceStorage.LAZY


def build_distance_store(problem: MaxDivProblem, resolved: DistanceStorage) -> DistanceStore:
    """Build the distance store for an already-resolved (non-AUTO) backend choice.

    The memory check guards only the cases that allocate a new matrix — computing it from vectors
    or expanding a condensed input; a square input is adopted as is.

    Raises:
        ValueError: For LAZY on a distance-input problem (no vectors to compute from), or when the
            full matrix cannot fit in physical memory at all.
    """
    match resolved:
        case DistanceStorage.FULL_MATRIX:
            if not problem.has_full_matrix:
                _check_full_matrix_fits_memory(problem)
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
        _check_full_matrix_fits_memory(problem)
        shared = SharedDistanceStore.allocate((problem.n, problem.n), KIND_FULL_MATRIX)
        problem.full_matrix(out=shared.buffer)
        return shared
    # The remaining cases, a full matrix as given or lazy, publish a built store; a lazy store holds
    # the vectors preprocessed for the metric, so its vector array is published — the segment must
    # hold what the distance reads expect.
    return publish_distance_store(build_distance_store(problem, resolved))


def _check_full_matrix_fits_memory(problem: MaxDivProblem) -> None:
    """Raise early when the problem's full matrix cannot fit in physical RAM at all."""
    check_fits_physical_memory(full_matrix_bytes(problem.n), lazy_available=isinstance(problem, VectorMaxDivProblem))
