"""The distance store factory base holds what the factory of every problem flavor shares."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager

from max_div._core.metrics._distance import DistanceMetric, DistanceStore

from .allocation import (
    DistanceStoreAllocator,
    InProcessDistanceStoreAllocator,
    SharedMemoryDistanceStoreAllocator,
)
from .shared_memory import SharedStoreSpec, attached_distance_store
from .storage import DistanceStorageType, DistanceStorageTypes


# ==================================================================================================
#  DistanceStoreFactory
# ==================================================================================================
class DistanceStoreFactory(ABC):
    """A distance store factory builds the distance stores that one solve reads, one per distance.

    Each problem flavor has a subclass, which decides what `AUTO` storage resolves to and builds the
    stores from that flavor's data.  The base class passes an explicitly chosen storage type through
    and places the stores in memory: in this process, or in shared memory for worker processes.

    "Storage type" names a `DistanceStorageType` value throughout; "kind" is reserved for
    `DistanceStore.kind`, the compiled selector that a distance store carries.
    """

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(self, storage_type: DistanceStorageType) -> None:
        """Keep the user's choice of storage type, possibly AUTO."""
        self._storage_type = storage_type

    @property
    @abstractmethod
    def distance_metrics(self) -> tuple[DistanceMetric | None, ...]:
        """Return the distance metric of each store, in store order; None for a store over given distances."""

    # --------------------------------------------------------------------------
    #  Policy
    # --------------------------------------------------------------------------
    def determine_storage_types(self) -> list[DistanceStorageType]:
        """Return the storage type of each store; an explicit choice passes through, the subclass decides AUTO."""
        if self._storage_type != DistanceStorageType.AUTO:
            return [self._storage_type] * len(self.distance_metrics)
        else:
            return self._determine_auto_storage_types()

    @abstractmethod
    def _determine_auto_storage_types(self) -> list[DistanceStorageType]:
        """Return the storage type that AUTO resolves to for each store, in store order."""

    def resolved_storage(self) -> DistanceStorageTypes:
        """Return each store's distance metric paired with its resolved storage type, in store order."""
        return DistanceStorageTypes(tuple(zip(self.distance_metrics, self.determine_storage_types(), strict=True)))

    # --------------------------------------------------------------------------
    #  Construction of the stores
    # --------------------------------------------------------------------------
    def create_stores(self) -> list[DistanceStore]:
        """Build the distance stores in this process, in store order.

        Raises:
            ValueError: When the subclass cannot build a store of the resolved storage type.
        """
        return self._build(InProcessDistanceStoreAllocator())

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

    @abstractmethod
    def _build(self, allocator: DistanceStoreAllocator) -> list[DistanceStore]:
        """Build every distance store through the given allocator, after the memory check the allocations need."""
