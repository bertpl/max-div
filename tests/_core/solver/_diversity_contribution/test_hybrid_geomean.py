import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._diversity_contribution import HybridGeoMeanTracker, SeparationTracker

from .helpers import selection_args

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 5
_VECTORS = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [6.0, 5.0], [10.0, 0.0]], dtype=np.float32)


@pytest.fixture
def term_trackers() -> tuple[SeparationTracker, SeparationTracker]:
    """Two separation trackers over the same points, one per distance metric."""
    return (
        SeparationTracker(DistanceStore.full_matrix_from_vectors(_VECTORS, DistanceMetric.l1_manhattan())),
        SeparationTracker(DistanceStore.full_matrix_from_vectors(_VECTORS, DistanceMetric.l2_euclidean())),
    )


def _geomean_of(arrays: list[np.ndarray]) -> np.ndarray:
    """Elementwise geometric mean of the arrays, in float64, as the reference."""
    return np.exp(np.mean(np.log(np.array(arrays, dtype=np.float64)), axis=0))


def _add_index_to_all_trackers(trackers, index: int) -> None:
    """Add point `index` to every tracker in `trackers`, in order."""
    for tracker in trackers:
        tracker.add(np.int32(index))


# =================================================================================================
#  Tests
# =================================================================================================
def test_rejects_fewer_than_two_term_trackers(term_trackers):
    """With a single term the objective is that term itself, so one term tracker is refused."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at least two"):
        HybridGeoMeanTracker([term_trackers[0]])


def test_empty_selection_combines_to_inf(term_trackers):
    """Every term tracker reports +inf for an empty selection, and so does their geometric mean."""
    # --- arrange ----------------------
    hybrid_tracker = HybridGeoMeanTracker(term_trackers)
    selected, n_selected = selection_args([], N)

    # --- act / assert -----------------
    assert np.all(np.isinf(hybrid_tracker.contribution_wrt_selection(selected, n_selected)))


@pytest.mark.parametrize("term_positions", [(0, 1), (0, 0, 1)], ids=["two_distances", "repeated_tracker"])
def test_selection_contribution_is_the_geomean_over_terms(term_trackers, term_positions: tuple[int, ...]):
    """The combined array is the elementwise geometric mean of the term trackers' arrays.

    A repeated tracker is counted once per term that reads it.
    """
    # --- arrange ----------------------
    hybrid_tracker = HybridGeoMeanTracker([term_trackers[position] for position in term_positions])
    _add_index_to_all_trackers((*term_trackers, hybrid_tracker), 0)
    _add_index_to_all_trackers((*term_trackers, hybrid_tracker), 3)
    selected, n_selected = selection_args([0, 3], N)

    # --- act --------------------------
    result = hybrid_tracker.contribution_wrt_selection(selected, n_selected)

    # --- assert -----------------------
    per_term = [term_trackers[position].contribution_wrt_selection(selected, n_selected) for position in term_positions]
    np.testing.assert_allclose(result, _geomean_of(per_term), rtol=1e-6)


def test_dataset_contributions_combine_the_term_trackers(term_trackers):
    """The dataset-wide array, and the array returned for a subset of indices, combine the term trackers' arrays."""
    # --- arrange ----------------------
    hybrid_tracker = HybridGeoMeanTracker(term_trackers)
    indices = np.array([4, 1], dtype=np.int32)

    # --- act --------------------------
    full_contribution = hybrid_tracker.contribution_wrt_dataset
    subset_contribution = hybrid_tracker.contribution_wrt_dataset_for(indices)

    # --- assert -----------------------
    expected = _geomean_of([tracker.contribution_wrt_dataset for tracker in term_trackers])
    np.testing.assert_allclose(full_contribution, expected, rtol=1e-6)
    np.testing.assert_allclose(subset_contribution, expected[indices], rtol=1e-6)


@pytest.mark.parametrize("mutation", ["add", "add_many", "remove", "remove_trial", "remove_many", "reset"])
def test_every_mutation_marks_the_combined_array_stale(term_trackers, mutation: str):
    """After any mutation the next read recombines the term trackers, never returning the array cached before it."""
    # --- arrange ----------------------
    hybrid_tracker = HybridGeoMeanTracker(term_trackers)
    _add_index_to_all_trackers((*term_trackers, hybrid_tracker), 0)
    selected, n_selected = selection_args([0], N)
    _ = hybrid_tracker.contribution_wrt_selection(selected, n_selected)  # fill the cache
    # the term trackers' selection changes (point 2 is added); the hybrid tracker is only told a mutation happened
    _add_index_to_all_trackers(term_trackers, 2)
    selected, n_selected = selection_args([0, 2], N)
    new_selection = np.array([0, 2], dtype=np.int32)

    # --- act --------------------------
    match mutation:
        case "add":
            hybrid_tracker.add(np.int32(2))
        case "add_many":
            hybrid_tracker.add_many(np.array([2], dtype=np.int32))
        case "remove":
            hybrid_tracker.remove(np.int32(2), new_selection)
        case "remove_trial":
            hybrid_tracker.remove_trial(np.int32(2), new_selection)
        case "remove_many":
            hybrid_tracker.remove_many(np.array([2], dtype=np.int32), new_selection)
        case "reset":
            hybrid_tracker.reset()
    after = hybrid_tracker.contribution_wrt_selection(selected, n_selected)

    # --- assert -----------------------
    expected = _geomean_of([tracker.contribution_wrt_selection(selected, n_selected) for tracker in term_trackers])
    np.testing.assert_allclose(after, expected, rtol=1e-6)


def test_restoring_pop_marks_the_combined_array_stale_and_plain_pop_does_not(term_trackers):
    """The term trackers hold the snapshots; a restoring pop changes their arrays, so the combined one is recombined."""
    # --- arrange ----------------------
    hybrid_tracker = HybridGeoMeanTracker(term_trackers)
    _add_index_to_all_trackers((*term_trackers, hybrid_tracker), 0)
    selected_0, n_0 = selection_args([0], N)
    at_snapshot = hybrid_tracker.contribution_wrt_selection(selected_0, n_0).copy()
    for tracker in (*term_trackers, hybrid_tracker):
        tracker.push_snapshot()
        tracker.push_snapshot()
    _add_index_to_all_trackers((*term_trackers, hybrid_tracker), 4)
    selected_04, n_04 = selection_args([0, 4], N)
    after_add = hybrid_tracker.contribution_wrt_selection(selected_04, n_04).copy()

    # --- act / assert -----------------
    for tracker in (*term_trackers, hybrid_tracker):
        tracker.pop_snapshot(restore=False)
    assert np.array_equal(hybrid_tracker.contribution_wrt_selection(selected_04, n_04), after_add)  # unchanged
    for tracker in (*term_trackers, hybrid_tracker):
        tracker.pop_snapshot(restore=True)
    np.testing.assert_allclose(hybrid_tracker.contribution_wrt_selection(selected_0, n_0), at_snapshot)


def test_store_is_not_defined_for_several_terms(term_trackers):
    """The terms may read different stores, so asking for one store raises."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="no single distance store"):
        _ = HybridGeoMeanTracker(term_trackers).store
