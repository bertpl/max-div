"""The distance store factory builds the distance stores for one solve.

Its input is the problem and the list of distance metrics that the diversity metric uses; its
output is one distance store per distance metric, in that order.  The factory owns three things:

- the policy that picks a storage type (full matrix or lazy) for each distance metric;
- the construction of each distance store, in this process or in shared memory;
- the rule for which distance stores share one array: every lazy distance store whose metric does
  not preprocess the vectors reads the user's raw vectors, so those stores share one array and one
  shared-memory segment; a metric that preprocesses gets an array of its own.

Preprocessing for a lazy distance store happens here; `compute_full_matrix` preprocesses the
vectors itself.
"""

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager

from max_div._core.metrics import DiversityObjective
from max_div._core.metrics._distance import (
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceMetric,
    DistanceStore,
    compute_full_matrix,
    expand_condensed,
    preprocess_vectors,
)
from max_div._core.problem import DistanceMaxDivProblem, MaxDivProblem, VectorMaxDivProblem

from .allocation import (
    DistanceStoreAllocator,
    InProcessDistanceStoreAllocator,
    SharedMemoryDistanceStoreAllocator,
)
from .memory_budget import AUTO_MEMORY_FRACTION, check_fits_physical_memory, full_matrix_bytes
from .shared_memory import SharedStoreSpec, attached_distance_store
from .storage import DistanceStorageType, DistanceStorageTypes

# The distance that a distance store holds: a distance metric over the problem's vectors, or None for
# the distances that a distance-input problem was given.
StoreDistance = DistanceMetric | None


# =================================================================================================
#  DistanceStoreFactory
# =================================================================================================
class DistanceStoreFactory:
    """The distance store factory builds the distance stores that one solve reads.

    The memory probe is injected, so that the storage-type policy is a pure function of its
    arguments and can be tested without the machine's RAM.

    "Storage type" names a `DistanceStorageType` value throughout; "kind" is reserved for
    `DistanceStore.kind`, the compiled selector that a distance store carries.
    """

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(
        self,
        problem: MaxDivProblem,
        distances: Sequence[StoreDistance],
        storage_type: DistanceStorageType,
        total_memory_bytes: int | None,
    ) -> None:
        """Bind the factory to the problem and the distances that it builds distance stores for.

        Args:
            problem: the problem whose vectors, or given distances, the distance stores hold.
            distances: one entry per distance store: a distance metric over the problem's vectors,
                or None for the given distances.  A distance-input problem accepts None only; a
                vector problem accepts distance metrics only.
            storage_type: the user's choice of storage type, possibly AUTO.
            total_memory_bytes: the total physical RAM of the machine, or None when it is unknown.

        Raises:
            ValueError: If an entry does not fit the problem flavor, or the list is empty.
        """
        if not distances:
            raise ValueError("A factory needs at least one distance.")
        is_vector_problem = isinstance(problem, VectorMaxDivProblem)
        for distance in distances:
            if not is_vector_problem and distance is not None:
                raise ValueError(
                    "A distance-input problem has no vectors; every entry must be None (its given distances)."
                )
        self._problem = problem
        self._distances = tuple(distances)
        self._storage_type = storage_type
        self._total_memory_bytes = total_memory_bytes

    @classmethod
    def for_objectives(
        cls,
        problem: MaxDivProblem,
        objectives: Sequence[DiversityObjective],
        storage_type: DistanceStorageType,
        total_memory_bytes: int | None,
    ) -> "DistanceStoreFactory":
        """Return the factory for the distinct distances the objectives read, one store per distance.

        The distances are kept as the objectives declare them (`None` for the problem's own), so a
        store can be looked up by the distance an objective declares, without resolving it. `None`
        is resolved to the problem's distance only when a store is actually built, and in the report.
        """
        return cls(problem, distinct_store_distances(objectives), storage_type, total_memory_bytes)

    @property
    def distances(self) -> tuple[StoreDistance, ...]:
        """Return the distances this factory builds stores for, in store order (as the objectives declare them)."""
        return self._distances

    # --------------------------------------------------------------------------
    #  Policy
    # --------------------------------------------------------------------------
    def determine_storage_types(self) -> list[DistanceStorageType]:
        """Return the storage type for each distance; an explicit choice passes through, AUTO is decided here.

        AUTO is decided differently for the two problem flavors, deliberately:

        - For a vector problem the distances are an internal artifact that the user never sees, so
          AUTO picks the full matrix when all the matrices together fit the memory fraction, and
          computes distances on demand otherwise.  This is one decision for all the distances.
        - For a distance-input problem the distances exist already, so AUTO stores them as a full
          matrix.

        When the total RAM is unknown, AUTO picks lazy, the one storage type that cannot page.
        """
        count = len(self._distances)
        if self._storage_type != DistanceStorageType.AUTO:
            return [self._storage_type] * count
        if not isinstance(self._problem, VectorMaxDivProblem):
            return [DistanceStorageType.FULL_MATRIX] * count
        if self._total_memory_bytes is None:
            return [DistanceStorageType.LAZY] * count
        if count * full_matrix_bytes(self._problem.n) <= self._total_memory_bytes * AUTO_MEMORY_FRACTION:
            return [DistanceStorageType.FULL_MATRIX] * count
        return [DistanceStorageType.LAZY] * count

    def resolved_storage(self) -> DistanceStorageTypes:
        """Return each store's distance paired with its resolved storage type, in store order.

        A `None` distance (the problem's own) is reported as the metric it resolves to, so the
        summary names it rather than leaving it blank.
        """
        return DistanceStorageTypes(tuple(zip(self._resolved_distances(), self.determine_storage_types(), strict=True)))

    def _resolved_distances(self) -> tuple[StoreDistance, ...]:
        """Return each store's distance with a vector problem's `None` (its own distance) replaced by its metric.

        A distance-input problem's `None` stays `None`: its default distance is `None` (it holds given
        distances, not a metric).
        """
        return tuple(distance or self._problem.default_distance_metric for distance in self._distances)

    # --------------------------------------------------------------------------
    #  Construction of the stores
    # --------------------------------------------------------------------------
    def create_stores(self) -> list[DistanceStore]:
        """Build the distance stores in this process, one per distance, in store order.

        Raises:
            ValueError: For the LAZY storage type on a distance-input problem, which has no vectors
                to compute distances from, or when the full matrices cannot fit in physical memory
                at all.
        """
        return self._build(InProcessDistanceStoreAllocator())

    def create_stores_by_distance(self) -> dict[StoreDistance, DistanceStore]:
        """Build the stores in this process, keyed by the distance each was built for (`None` for the problem's own)."""
        return dict(zip(self._distances, self.create_stores(), strict=True))

    @contextmanager
    def publish_distance_stores(self) -> Iterator[tuple[SharedStoreSpec, ...]]:
        """Build the distance stores in shared memory and yield their specs, for the duration of the block.

        This is a context manager.  Inside the block the shared-memory segments exist and worker
        processes can attach to them with the yielded specs, through `attach_distance_stores`.  On
        exit the segments are destroyed, so leave the block only after every worker is done.

        Raises:
            ValueError: as `create_stores`.
        """
        allocator = SharedMemoryDistanceStoreAllocator()
        try:
            self._build(allocator)
            yield allocator.specs
        finally:
            allocator.close()

    @classmethod
    @contextmanager
    def attach_distance_stores(cls, specs: Sequence[SharedStoreSpec]) -> Iterator[list[DistanceStore]]:
        """Yield the distance stores that the specs describe, read from their segments, for the duration of the block.

        This is the worker-side counterpart of `publish_distance_stores`.  On exit every mapping is
        closed; no segment is destroyed, because the segments belong to the process that published
        them.
        """
        with ExitStack() as stack:
            yield [stack.enter_context(attached_distance_store(spec)) for spec in specs]

    def _build(self, allocator: DistanceStoreAllocator) -> list[DistanceStore]:
        """Build every distance store through the given allocator, after the memory check the allocations need."""
        store_types = self.determine_storage_types()
        problem = self._problem
        n = problem.n
        if isinstance(problem, DistanceMaxDivProblem):
            if DistanceStorageType.LAZY in store_types:
                raise ValueError(
                    "Lazy distance storage computes distances from vectors, which a distance-input "
                    "problem does not have; choose FULL_MATRIX, or construct the problem from vectors."
                )
            if not problem.has_full_matrix:
                check_fits_physical_memory(len(store_types) * full_matrix_bytes(n), lazy_available=False)
            return [self._store_over_given_distances(problem, allocator) for _ in store_types]
        if not isinstance(problem, VectorMaxDivProblem):  # pragma: no cover -- the two flavors above are the only ones
            raise TypeError(f"Unknown problem flavor {type(problem).__name__}.")
        n_full = sum(store_type == DistanceStorageType.FULL_MATRIX for store_type in store_types)
        if n_full:
            check_fits_physical_memory(n_full * full_matrix_bytes(n), lazy_available=True)
        stores = []
        for distance, store_type in zip(self._resolved_distances(), store_types, strict=True):
            assert distance is not None  # noqa: S101 -- a vector problem always resolves to a metric
            if store_type == DistanceStorageType.FULL_MATRIX:
                matrix = allocator.allocate((n, n), KIND_FULL_MATRIX)
                compute_full_matrix(problem.vectors, distance, out=matrix)
                stores.append(DistanceStore.full_matrix(matrix))
            else:
                # preprocess_vectors returns problem.vectors itself for a metric that does not preprocess,
                # so the shared-memory allocator sees one array and puts it in one segment
                adopted = allocator.adopt(preprocess_vectors(problem.vectors, distance), KIND_LAZY, distance)
                stores.append(DistanceStore.lazy(adopted, distance))
        return stores

    @staticmethod
    def _store_over_given_distances(problem: DistanceMaxDivProblem, allocator: DistanceStoreAllocator) -> DistanceStore:
        """Return the full-matrix distance store over the given distances; a condensed input is expanded."""
        n = problem.n
        if problem.has_full_matrix:
            matrix = allocator.adopt(problem.distances, KIND_FULL_MATRIX, None)
        else:
            matrix = allocator.allocate((n, n), KIND_FULL_MATRIX)
            expand_condensed(problem.distances, n, out=matrix)
        return DistanceStore.full_matrix(matrix)


# ==================================================================================================
#  Helpers
# ==================================================================================================
def distinct_store_distances(objectives: Sequence[DiversityObjective]) -> tuple[StoreDistance, ...]:
    """Return the distinct distances the objectives read, in first-seen order (`None` for the problem's own).

    The order is deterministic from the objectives alone, so a worker process rebuilds the same
    one-store-per-distance mapping from its objectives and the stores it attaches.
    """
    return tuple(
        dict.fromkeys(distance for objective in objectives for distance in objective.distinct_distance_metrics())
    )
