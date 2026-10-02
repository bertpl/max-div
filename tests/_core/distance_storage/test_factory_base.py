import numpy as np
import pytest

from max_div._core.distance_storage import DistanceStorageType, DistanceStoreFactory, attached_distance_store
from max_div._core.distance_storage.allocation import DistanceStoreAllocator
from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import KIND_FULL_MATRIX, DistanceStore

# ==================================================================================================
#  Fixtures / helpers
# ==================================================================================================
L1 = DistanceMetric.l1_manhattan()
L2 = DistanceMetric.l2_euclidean()


class _TwoStoreFactory(DistanceStoreFactory):
    """A minimal factory with 2 stores, each a 3 by 3 matrix of ones, whose AUTO is the full matrix."""

    @property
    def distance_metrics(self) -> tuple[DistanceMetric, ...]:
        """Return the 2 metrics that the stores are reported under."""
        return (L1, L2)

    def _determine_auto_storage_types(self) -> list[DistanceStorageType]:
        """Return the full matrix for both stores."""
        return [DistanceStorageType.FULL_MATRIX, DistanceStorageType.FULL_MATRIX]

    def _build(self, allocator: DistanceStoreAllocator) -> list[DistanceStore]:
        """Build both stores through the allocator, whatever the resolved storage type."""
        stores = []
        for _ in self.distance_metrics:
            matrix = allocator.allocate((3, 3), KIND_FULL_MATRIX)
            matrix[:] = 1.0
            stores.append(DistanceStore.full_matrix(matrix))
        return stores


# ==================================================================================================
#  Policy
# ==================================================================================================
@pytest.mark.parametrize(
    "storage, expected",
    [
        (DistanceStorageType.LAZY, DistanceStorageType.LAZY),  # an explicit choice passes through
        (DistanceStorageType.AUTO, DistanceStorageType.FULL_MATRIX),  # AUTO is the subclass's decision
    ],
)
def test_storage_types_are_the_explicit_choice_or_the_subclass_decision(
    storage: DistanceStorageType, expected: DistanceStorageType
):
    """An explicit storage type applies to every store; AUTO resolves to what the subclass decides."""
    # --- arrange ----------------------
    factory = _TwoStoreFactory(storage)

    # --- act / assert -----------------
    assert factory.determine_storage_types() == [expected, expected]
    assert factory.resolved_storage().per_store == ((L1, expected), (L2, expected))


# ==================================================================================================
#  Placing the stores in memory
# ==================================================================================================
def test_published_stores_are_read_back_in_store_order():
    """A worker that attaches to the published specs reads one store per spec, with the published contents."""
    # --- arrange ----------------------
    factory = _TwoStoreFactory(DistanceStorageType.AUTO)

    # --- act --------------------------
    with factory.publish_distance_stores() as specs, DistanceStoreFactory.attach_distance_stores(specs) as attached:
        matrices = [np.array(store.matrix) for store in attached]

    # --- assert -----------------------
    assert len(factory.create_stores()) == len(matrices) == 2
    for matrix in matrices:
        np.testing.assert_array_equal(matrix, np.ones((3, 3), dtype=np.float32))


def test_leaving_the_publish_block_destroys_the_segments():
    """After the block the segments are gone, so an attach with a stale spec fails instead of reading freed memory."""
    # --- arrange ----------------------
    with _TwoStoreFactory(DistanceStorageType.AUTO).publish_distance_stores() as specs:
        stale = specs[0]

    # --- act / assert -----------------
    with pytest.raises(FileNotFoundError), attached_distance_store(stale):
        pass
