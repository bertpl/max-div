"""The factory is the one place a problem and the distances a diversity metric reads become a set of distance stores.

It owns three things:

- the policy that picks a storage type per distance;
- the construction of each store, in process or in shared memory;
- the rule for which stores share one array: lazy stores over metrics that do not preprocess share the
  user's raw vectors, one array and one segment; a preprocessing metric gets its own.

Preprocessing for a lazy store happens here; `compute_full_matrix` preprocesses the vectors itself.
"""

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager

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

from .allocation import InProcessAllocator, SharedMemoryAllocator, StoreAllocator
from .memory_budget import AUTO_MEMORY_FRACTION, check_fits_physical_memory, full_matrix_bytes
from .shared_memory import SharedStoreSpec, attached_distance_store
from .storage import DistanceStorageType

# A distance a store reads: a metric over the problem's vectors, or None for the distances a
# distance-input problem was given.
StoreDistance = DistanceMetric | None


# =================================================================================================
#  DistanceStoreFactory
# =================================================================================================
class DistanceStoreFactory:
    """Build the distance stores one solve reads, from the problem and the distances the diversity metric uses.

    The memory probe is injected so the policy is a pure function of its arguments and testable
    without the machine's RAM.

    "Storage type" names a `DistanceStorageType` value throughout; "kind" is reserved for
    `DistanceStore.kind`, the compiled selector a store carries.
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
        """Bind the factory to what it builds from.

        Args:
            problem: the problem whose vectors, or given distances, the stores hold.
            distances: one entry per store: a metric over the problem's vectors, or None for the
                given distances.  A distance-input problem accepts None only; a vector problem
                accepts metrics only.
            storage_type: the user's choice, possibly AUTO.
            total_memory_bytes: total physical RAM, or None when unknown.

        Raises:
            ValueError: If an entry does not fit the problem flavor, or the list is empty.
        """
        if not distances:
            raise ValueError("A factory needs at least one distance.")
        is_vector_problem = isinstance(problem, VectorMaxDivProblem)
        for distance in distances:
            if is_vector_problem and distance is None:
                raise ValueError("A vector problem has no given distances; every entry must be a DistanceMetric.")
            if not is_vector_problem and distance is not None:
                raise ValueError(
                    "A distance-input problem has no vectors; every entry must be None (its given distances)."
                )
        self._problem = problem
        self._distances = tuple(distances)
        self._storage_type = storage_type
        self._total_memory_bytes = total_memory_bytes

    @classmethod
    def for_problem(
        cls, problem: MaxDivProblem, storage_type: DistanceStorageType, total_memory_bytes: int | None
    ) -> "DistanceStoreFactory":
        """Return the one-distance factory: over the vector problem's metric, or over the given distances."""
        distance = problem.distance_metric if isinstance(problem, VectorMaxDivProblem) else None
        return cls(problem, [distance], storage_type, total_memory_bytes)

    # --------------------------------------------------------------------------
    #  Policy
    # --------------------------------------------------------------------------
    def determine_storage_types(self) -> list[DistanceStorageType]:
        """Return the resolved storage type per distance; explicit choices pass through, AUTO is decided here.

        AUTO semantics differ per problem flavor, deliberately:

        - Vector problems: distances are an internal artifact the user never sees, so AUTO picks
          the full matrix when every matrix together fits the memory fraction, and computes
          distances on demand otherwise — one decision for the whole set.
        - Distance problems: the distances exist already, so AUTO stores them as a full matrix.

        An unknown RAM total degrades to lazy, the one storage type that cannot page.
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

    # --------------------------------------------------------------------------
    #  Construction of the stores
    # --------------------------------------------------------------------------
    def create_stores(self) -> list[DistanceStore]:
        """Build the stores in this process, one per distance in order.

        Raises:
            ValueError: For LAZY on a distance-input problem (no vectors to compute from), or when
                the full matrices cannot fit in physical memory at all.
        """
        return self._build(InProcessAllocator())

    def create_shared_stores(self) -> "SharedStoreSet":
        """Build the stores in shared memory, for worker processes to attach to.

        Raises:
            ValueError: as `create_stores`.
        """
        allocator = SharedMemoryAllocator()
        try:
            stores = self._build(allocator)
        except BaseException:
            allocator.close()
            raise
        return SharedStoreSet(allocator, stores)

    @classmethod
    @contextmanager
    def attach_stores(cls, specs: Sequence[SharedStoreSpec]) -> Iterator[list[DistanceStore]]:
        """Yield the stores a `SharedStoreSet` published, read from its segments, for the duration of the block.

        `attach_stores` inverts `create_shared_stores` in the attaching process.  Every mapping is
        closed on exit and no segment is unlinked: they belong to the publisher.
        """
        with ExitStack() as stack:
            yield [stack.enter_context(attached_distance_store(spec)) for spec in specs]

    def _build(self, allocator: StoreAllocator) -> list[DistanceStore]:
        """Build every store through the given allocator, after the memory check the allocations need."""
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
        for distance, store_type in zip(self._distances, store_types, strict=True):
            assert distance is not None  # noqa: S101 -- the constructor rejects None for a vector problem
            if store_type == DistanceStorageType.FULL_MATRIX:
                matrix = allocator.allocate((n, n), KIND_FULL_MATRIX)
                compute_full_matrix(problem.vectors, distance, out=matrix)
                stores.append(DistanceStore.full_matrix(matrix))
            else:
                # preprocess_vectors returns problem.vectors itself for a metric that does not preprocess,
                # so the shared-memory allocator sees one array and publishes it once
                adopted = allocator.adopt(preprocess_vectors(problem.vectors, distance), KIND_LAZY, distance)
                stores.append(DistanceStore.lazy(adopted, distance))
        return stores

    @staticmethod
    def _store_over_given_distances(problem: DistanceMaxDivProblem, allocator: StoreAllocator) -> DistanceStore:
        """Return the full-matrix store over a distance-input problem's distances, expanding a condensed input."""
        n = problem.n
        if problem.has_full_matrix:
            matrix = allocator.adopt(problem.distances, KIND_FULL_MATRIX, None)
        else:
            matrix = allocator.allocate((n, n), KIND_FULL_MATRIX)
            expand_condensed(problem.distances, n, out=matrix)
        return DistanceStore.full_matrix(matrix)


# =================================================================================================
#  SharedStoreSet
# =================================================================================================
class SharedStoreSet:
    """A set holds the stores a factory built into shared memory, their segments, and the specs a worker attaches with.

    Closing the set closes its allocator (see `SharedMemoryAllocator.close`).
    """

    def __init__(self, allocator: SharedMemoryAllocator, stores: list[DistanceStore]) -> None:
        """Hold the allocator that owns the segments, and the stores built over them."""
        self._allocator = allocator
        self._stores = stores

    @property
    def stores(self) -> list[DistanceStore]:
        """Return the stores, in the factory's distance order, reading the segments in this process."""
        return self._stores

    @property
    def specs(self) -> tuple[SharedStoreSpec, ...]:
        """Return what `DistanceStoreFactory.attach_stores` needs to rebuild the stores elsewhere."""
        return self._allocator.specs

    def close(self) -> None:
        """Close the allocator (see `SharedMemoryAllocator.close`) and drop the stores."""
        self._stores = []
        self._allocator.close()

    def __enter__(self) -> "SharedStoreSet":
        """Return the set itself, so the segments are scoped to a `with` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the set."""
        self.close()
