"""The distance store factory of a vector problem computes each store's distances from the vectors."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics._distance import (
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceMetric,
    DistanceStore,
    compute_full_matrix,
)

from .allocation import DistanceStoreAllocator
from .factory_base import DistanceStoreFactory
from .memory_budget import (
    AUTO_MEMORY_FRACTION,
    check_fits_physical_memory,
    full_matrix_bytes,
)
from .storage import DistanceStorageType


# ==================================================================================================
#  VectorProblemDistanceStoreFactory
# ==================================================================================================
class VectorProblemDistanceStoreFactory(DistanceStoreFactory):
    """This factory builds one distance store per distance metric over a vector problem's vectors.

    A store is a full matrix that is computed up front, or a lazy store that computes each distance
    on demand.

    Every lazy store whose metric does not preprocess the vectors reads the user's raw vectors, so
    those stores share one array and one shared-memory segment; a metric that preprocesses gets an
    array of its own.

    The machine's total RAM is passed in as `total_memory_bytes`, so that the storage-type policy
    depends only on the constructor arguments and can be tested without probing the machine's RAM.
    """

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(
        self,
        vectors: NDArray[np.float32],
        distance_metrics: Sequence[DistanceMetric],
        storage_type: DistanceStorageType,
        total_memory_bytes: int | None,
    ) -> None:
        """Bind the factory to the vectors and to one distance metric per distance store.

        Args:
            vectors: the problem's vectors, one row per item.
            distance_metrics: one distance metric per distance store.
            storage_type: the user's choice of storage type, possibly AUTO.
            total_memory_bytes: the total physical RAM of the machine, or None when it is unknown.

        Raises:
            ValueError: If the list of distance metrics is empty.
        """
        if not distance_metrics:
            raise ValueError("A factory needs at least one distance metric.")
        super().__init__(storage_type)
        self._vectors = vectors
        self._distance_metrics = tuple(distance_metrics)
        self._total_memory_bytes = total_memory_bytes

    @property
    def distance_metrics(self) -> tuple[DistanceMetric, ...]:
        """Return the distance metric of each store, in store order."""
        return self._distance_metrics

    # --------------------------------------------------------------------------
    #  Policy
    # --------------------------------------------------------------------------
    def _determine_auto_storage_types(self) -> list[DistanceStorageType]:
        """Return `FULL_MATRIX` for all stores when their full matrices fit in the memory budget, else `LAZY`.

        The memory budget is `AUTO_MEMORY_FRACTION` of the total RAM.

        The distances of a vector problem are an internal artifact that the user never sees, so AUTO
        is free to compute them on demand.  AUTO makes one decision for all the stores.

        When the total RAM is unknown, AUTO picks lazy, because a lazy store holds only the vectors
        and so cannot force the machine to page to disk.
        """
        n_stores = len(self._distance_metrics)
        if self._total_memory_bytes is None:
            return [DistanceStorageType.LAZY] * n_stores
        elif n_stores * full_matrix_bytes(self._n) <= self._total_memory_bytes * AUTO_MEMORY_FRACTION:
            return [DistanceStorageType.FULL_MATRIX] * n_stores
        else:
            return [DistanceStorageType.LAZY] * n_stores

    # --------------------------------------------------------------------------
    #  Construction of the stores
    # --------------------------------------------------------------------------
    def _build(self, allocator: DistanceStoreAllocator) -> list[DistanceStore]:
        """Build every distance store through the given allocator.

        Raises:
            ValueError: When the full matrices cannot fit in physical memory at all.
        """
        storage_types = self.determine_storage_types()
        n = self._n
        n_full = sum(storage_type == DistanceStorageType.FULL_MATRIX for storage_type in storage_types)
        if n_full:
            check_fits_physical_memory(n_full * full_matrix_bytes(n), lazy_available=True)
        stores = []
        for distance_metric, storage_type in zip(self._distance_metrics, storage_types, strict=True):
            if storage_type == DistanceStorageType.FULL_MATRIX:
                matrix = allocator.allocate((n, n), KIND_FULL_MATRIX)
                compute_full_matrix(self._vectors, distance_metric, out=matrix)
                stores.append(DistanceStore.full_matrix(matrix))
            else:
                # `DistanceMetric.preprocess` returns the vectors themselves for a metric that does not preprocess,
                # so the shared-memory allocator sees one array and puts it in one segment
                preprocessed_vectors = allocator.adopt(
                    distance_metric.preprocess(self._vectors), KIND_LAZY, distance_metric
                )
                stores.append(DistanceStore.lazy(preprocessed_vectors, distance_metric))
        return stores

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @property
    def _n(self) -> int:
        """Return the number of items, which is the number of rows of the vectors."""
        return self._vectors.shape[0]
