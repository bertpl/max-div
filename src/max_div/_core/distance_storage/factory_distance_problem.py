"""The distance store factory of a distance-input problem stores the given distances as a full matrix."""

import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics._distance import KIND_FULL_MATRIX, DistanceStore, expand_condensed

from .allocation import DistanceStoreAllocator
from .factory_base import DistanceStoreFactory
from .memory_budget import check_fits_physical_memory, full_matrix_bytes
from .storage import DistanceStorageType


# ==================================================================================================
#  DistanceProblemDistanceStoreFactory
# ==================================================================================================
class DistanceProblemDistanceStoreFactory(DistanceStoreFactory):
    """This factory builds the one distance store of a distance-input problem, over its given distances.

    The given distances come without a metric, so the store is always a full matrix: a square input
    is read as given, and a condensed input is expanded.
    """

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(self, distances: NDArray[np.float32], n: int, storage_type: DistanceStorageType) -> None:
        """Bind the factory to the given distances.

        Args:
            distances: the problem's distances as provided: a square (n, n) matrix or a condensed 1D vector.
            n: the number of items.
            storage_type: the user's choice of storage type, possibly AUTO.
        """
        super().__init__(storage_type)
        self._distances = distances
        self._n = n

    @property
    def distance_metrics(self) -> tuple[None]:
        """Return the single entry of the one store, which is None: the given distances have no metric."""
        return (None,)

    # --------------------------------------------------------------------------
    #  Policy
    # --------------------------------------------------------------------------
    def _determine_auto_storage_types(self) -> list[DistanceStorageType]:
        """Return the full matrix: the distances exist already, so AUTO stores them as they are."""
        return [DistanceStorageType.FULL_MATRIX]

    # --------------------------------------------------------------------------
    #  Construction of the stores
    # --------------------------------------------------------------------------
    def _build(self, allocator: DistanceStoreAllocator) -> list[DistanceStore]:
        """Build the full-matrix distance store over the given distances; a condensed input is expanded.

        Raises:
            ValueError: For the LAZY storage type, which has no vectors to compute distances from, or
                when the expanded matrix cannot fit in physical memory at all.
        """
        if DistanceStorageType.LAZY in self.determine_storage_types():
            raise ValueError(
                "Lazy distance storage computes distances from vectors, which a distance-input "
                "problem does not have; choose FULL_MATRIX, or construct the problem from vectors."
            )
        if self._distances.ndim == 2:
            matrix = allocator.adopt(self._distances, KIND_FULL_MATRIX, None)
        else:
            check_fits_physical_memory(full_matrix_bytes(self._n), lazy_available=False)
            matrix = allocator.allocate((self._n, self._n), KIND_FULL_MATRIX)
            expand_condensed(self._distances, self._n, out=matrix)
        return [DistanceStore.full_matrix(matrix)]
