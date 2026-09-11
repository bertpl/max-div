import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.solver._distance_storage import DistanceStorageType, DistanceStorageTypes


def test_storage_values_are_the_labels_users_see():
    """The enum values are the strings the solution summary reports and users may pass."""
    # --- act / assert -----------------
    assert [storage.value for storage in DistanceStorageType] == ["auto", "full_matrix", "lazy"]


@pytest.mark.parametrize(
    "per_store, expected",
    [
        (
            (
                (DistanceMetric.l1_manhattan(), DistanceStorageType.FULL_MATRIX),
                (DistanceMetric.l2_euclidean(), DistanceStorageType.FULL_MATRIX),
                (DistanceMetric.geometric_mean(), DistanceStorageType.LAZY),
            ),
            "full_matrix (L1, L2), lazy (geomean)",
        ),
        (((None, DistanceStorageType.FULL_MATRIX),), "full_matrix"),
        ((), ""),
    ],
    ids=["grouped-by-type", "given-distances-omit-metric", "empty"],
)
def test_summary_groups_distances_by_storage_type(per_store, expected):
    """Group distances by storage type; a distance-input store shows its type alone; empty renders nothing."""
    # --- act / assert -----------------
    assert str(DistanceStorageTypes(per_store)) == expected
