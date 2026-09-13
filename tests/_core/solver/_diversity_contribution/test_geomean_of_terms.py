import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._diversity_contribution import GeoMeanOfTermsTracker, SeparationTracker

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 5
_VECTORS = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [6.0, 5.0], [10.0, 0.0]], dtype=np.float32)


@pytest.fixture
def members() -> tuple[SeparationTracker, SeparationTracker]:
    """Two separation trackers over the same points, one per distance metric."""
    return (
        SeparationTracker(DistanceStore.full_matrix_from_vectors(_VECTORS, DistanceMetric.l1_manhattan())),
        SeparationTracker(DistanceStore.full_matrix_from_vectors(_VECTORS, DistanceMetric.l2_euclidean())),
    )


def _selection_args(indices: list[int]) -> tuple[np.ndarray, np.int32]:
    """Build the (selected, n_selected) argument pair for contribution reads from a list of selected indices."""
    selected = np.full(N, False, dtype=np.bool)
    selected[indices] = True
    return selected, np.int32(len(indices))


def _geomean_of(arrays: list[np.ndarray]) -> np.ndarray:
    """Elementwise geometric mean of the arrays, in float64, as the reference."""
    return np.exp(np.mean(np.log(np.array(arrays, dtype=np.float64)), axis=0))


def _add_to_all(trackers, index: int) -> None:
    for tracker in trackers:
        tracker.add(np.int32(index))


# =================================================================================================
#  Tests
# =================================================================================================
def test_rejects_fewer_than_two_members(members):
    """A one-term geometric mean is the term's own tracker, so one member is refused."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at least two"):
        GeoMeanOfTermsTracker([members[0]])


def test_selection_contribution_is_the_geomean_over_terms(members):
    """The combined array is the elementwise geometric mean of the members' arrays, +inf where all members are +inf."""
    # --- arrange ----------------------
    combined = GeoMeanOfTermsTracker(members)
    selected, n_selected = _selection_args([])
    assert np.all(np.isinf(combined.contribution_wrt_selection(selected, n_selected)))  # empty selection
    _add_to_all((*members, combined), 0)
    _add_to_all((*members, combined), 3)
    selected, n_selected = _selection_args([0, 3])

    # --- act --------------------------
    result = combined.contribution_wrt_selection(selected, n_selected)

    # --- assert -----------------------
    expected = _geomean_of([member.contribution_wrt_selection(selected, n_selected) for member in members])
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_a_tracker_read_by_two_terms_counts_twice(members):
    """Members are one per term, so a repeated member enters the mean once per term it serves."""
    # --- arrange ----------------------
    l1, l2 = members
    combined = GeoMeanOfTermsTracker([l1, l1, l2])
    _add_to_all((l1, l2), 0)
    selected, n_selected = _selection_args([0])

    # --- act --------------------------
    result = combined.contribution_wrt_selection(selected, n_selected)

    # --- assert -----------------------
    a, b = l1.contribution_wrt_selection(selected, n_selected), l2.contribution_wrt_selection(selected, n_selected)
    np.testing.assert_allclose(result, _geomean_of([a, a, b]), rtol=1e-6)


def test_dataset_contributions_combine_the_members(members):
    """The dataset-wide array and its targeted read are the geometric mean of the members' dataset-wide arrays."""
    # --- arrange ----------------------
    combined = GeoMeanOfTermsTracker(members)
    indices = np.array([4, 1], dtype=np.int32)

    # --- act --------------------------
    full = combined.contribution_wrt_dataset
    targeted = combined.contribution_wrt_dataset_for(indices)

    # --- assert -----------------------
    expected = _geomean_of([member.contribution_wrt_dataset for member in members])
    np.testing.assert_allclose(full, expected, rtol=1e-6)
    np.testing.assert_allclose(targeted, expected[indices], rtol=1e-6)


@pytest.mark.parametrize("mutation", ["add", "add_many", "remove", "remove_trial", "remove_many", "reset"])
def test_every_mutation_marks_the_combined_array_stale(members, mutation: str):
    """After any mutation the next read recombines the members, so it never returns the array from before."""
    # --- arrange ----------------------
    combined = GeoMeanOfTermsTracker(members)
    _add_to_all((*members, combined), 0)
    selected, n_selected = _selection_args([0])
    before = combined.contribution_wrt_selection(selected, n_selected).copy()
    # the members move on (point 2 joins the selection); the combined tracker is only told a mutation happened
    _add_to_all(members, 2)
    selected, n_selected = _selection_args([0, 2])
    new_selection = np.array([0, 2], dtype=np.int32)
    assert np.array_equal(combined.contribution_wrt_selection(selected, n_selected), before)  # stale until told

    # --- act --------------------------
    match mutation:
        case "add":
            combined.add(np.int32(2))
        case "add_many":
            combined.add_many(np.array([2], dtype=np.int32))
        case "remove":
            combined.remove(np.int32(2), new_selection)
        case "remove_trial":
            combined.remove_trial(np.int32(2), new_selection)
        case "remove_many":
            combined.remove_many(np.array([2], dtype=np.int32), new_selection)
        case "reset":
            combined.reset()
    after = combined.contribution_wrt_selection(selected, n_selected)

    # --- assert -----------------------
    expected = _geomean_of([member.contribution_wrt_selection(selected, n_selected) for member in members])
    np.testing.assert_allclose(after, expected, rtol=1e-6)


def test_restoring_pop_marks_the_combined_array_stale_and_plain_pop_does_not(members):
    """The members hold the snapshots; a restoring pop changes their arrays, so the combined one is recombined."""
    # --- arrange ----------------------
    combined = GeoMeanOfTermsTracker(members)
    _add_to_all((*members, combined), 0)
    selected_0, n_0 = _selection_args([0])
    at_snapshot = combined.contribution_wrt_selection(selected_0, n_0).copy()
    for tracker in (*members, combined):
        tracker.push_snapshot()
        tracker.push_snapshot()
    _add_to_all((*members, combined), 4)
    selected_04, n_04 = _selection_args([0, 4])
    after_add = combined.contribution_wrt_selection(selected_04, n_04).copy()

    # --- act / assert -----------------
    for tracker in (*members, combined):
        tracker.pop_snapshot(restore=False)
    assert np.array_equal(combined.contribution_wrt_selection(selected_04, n_04), after_add)  # unchanged
    for tracker in (*members, combined):
        tracker.pop_snapshot(restore=True)
    np.testing.assert_allclose(combined.contribution_wrt_selection(selected_0, n_0), at_snapshot)


def test_copy_is_independent(members):
    """A copy combines its own copies of the members, so mutating the original leaves it unchanged."""
    # --- arrange ----------------------
    combined = GeoMeanOfTermsTracker(members)
    _add_to_all((*members, combined), 0)
    selected_0, n_0 = _selection_args([0])
    copied = combined.copy()
    before = copied.contribution_wrt_selection(selected_0, n_0).copy()

    # --- act --------------------------
    _add_to_all((*members, combined), 1)

    # --- assert -----------------------
    assert len(copied.term_trackers) == 2 and all(
        c is not m for c, m in zip(copied.term_trackers, members, strict=True)
    )
    np.testing.assert_array_equal(copied.contribution_wrt_selection(selected_0, n_0), before)


def test_store_is_not_defined_for_several_terms(members):
    """The terms may read different stores, so asking for one store raises."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="no single distance store"):
        _ = GeoMeanOfTermsTracker(members).store
