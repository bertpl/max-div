import pytest

from max_div._core.distance_storage import DistanceStorageType, DistanceStorageTypes


def test_storage_values_are_the_labels_users_see():
    """The enum values are the strings the solution summary reports and users may pass."""
    # --- act / assert -----------------
    assert [storage.value for storage in DistanceStorageType] == ["auto", "full_matrix", "lazy"]


@pytest.mark.parametrize(
    "per_store, expected",
    [
        (
            (
                ("L1", DistanceStorageType.FULL_MATRIX),
                ("L2", DistanceStorageType.FULL_MATRIX),
                ("geomean", DistanceStorageType.LAZY),
            ),
            "full_matrix (L1, L2), lazy (geomean)",
        ),
        ((("user distances", DistanceStorageType.FULL_MATRIX),), "full_matrix (user distances)"),
        ((), ""),
    ],
    ids=["grouped-by-type", "given-distances", "empty"],
)
def test_summary_groups_labels_by_storage_type(per_store, expected):
    """Group the store labels by storage type; empty renders nothing."""
    # --- act / assert -----------------
    assert str(DistanceStorageTypes(per_store)) == expected
