from max_div._core.solver._distance_storage import DistanceStorage


def test_storage_values_are_the_labels_users_see():
    """The enum values are the strings the solution summary reports and users may pass."""
    # --- act / assert -----------------
    assert [storage.value for storage in DistanceStorage] == ["auto", "full_matrix", "lazy"]
