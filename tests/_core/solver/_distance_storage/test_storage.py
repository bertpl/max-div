from max_div._core.metrics import DistanceMetric
from max_div._core.solver._distance_storage import DistanceStorageType, DistanceStorageTypes


def test_storage_values_are_the_labels_users_see():
    """The enum values are the strings the solution summary reports and users may pass."""
    # --- act / assert -----------------
    assert [storage.value for storage in DistanceStorageType] == ["auto", "full_matrix", "lazy"]


def test_summary_groups_metrics_by_storage_type():
    """The summary groups distances by storage type and lists each type's metric labels."""
    # --- arrange ----------------------
    types = DistanceStorageTypes(
        (
            (DistanceMetric.l1_manhattan(), DistanceStorageType.FULL_MATRIX),
            (DistanceMetric.l2_euclidean(), DistanceStorageType.FULL_MATRIX),
            (DistanceMetric.geometric_mean(), DistanceStorageType.LAZY),
        )
    )

    # --- act / assert -----------------
    assert str(types) == "full_matrix (L1, L2), lazy (geomean)"


def test_summary_of_a_given_distances_store_omits_the_metric():
    """A distance-input store has no metric, so its type stands alone."""
    # --- act / assert -----------------
    assert str(DistanceStorageTypes(((None, DistanceStorageType.FULL_MATRIX),))) == "full_matrix"


def test_empty_summary_is_blank():
    """An unreported layout renders nothing."""
    # --- act / assert -----------------
    assert str(DistanceStorageTypes()) == ""
