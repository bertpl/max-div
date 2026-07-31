import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.solver._diversity_contribution import SeparationTracker
from max_div._core.solver._diversity_contribution._separation import (
    compute_separation,
    update_separation_add,
    update_separation_remove,
)


# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
@pytest.fixture
def tracker() -> SeparationTracker:
    vectors = np.array([[0.0], [1.0], [3.0], [6.0], [10.0]], dtype=np.float32)
    store = DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=vectors.shape[0])
    return SeparationTracker(store)


def _selection_args(indices: list[int], n: int) -> tuple[np.ndarray, np.int32]:
    """Build the (selected, n_selected) argument pair for contribution reads from a list of selected indices."""
    selected = np.full(n, False, dtype=np.bool)
    selected[indices] = True
    return selected, np.int32(len(indices))


# =================================================================================================
#  Tests
# =================================================================================================
def test_construction_fresh(tracker: SeparationTracker):
    # --- arrange -----------------------------------------
    selected, n_selected = _selection_args([], 5)

    # --- assert ------------------------------------------
    # global contribution: nearest-neighbor distances of points [0, 1, 3, 6, 10] on a line
    np.testing.assert_allclose(tracker.contribution_wrt_dataset, [1, 1, 2, 3, 4])
    # empty selection: all contributions +inf
    assert np.all(np.isinf(tracker.contribution_wrt_selection(selected, n_selected)))


def test_construction_precomputed_arrays_skip_recompute():
    # --- arrange -----------------------------------------
    vectors = np.array([[0.0], [1.0], [5.0]], dtype=np.float32)
    store = DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=3)
    sep_global = compute_separation(store)
    sep_selected = np.array([7.0, 8.0, 9.0], dtype=np.float32)

    # --- act ---------------------------------------------
    tracker = SeparationTracker(store, sep_global=sep_global, sep_selected=sep_selected)

    # --- assert ------------------------------------------
    assert tracker.contribution_wrt_dataset is sep_global  # taken as-is, not recomputed
    selected, n_selected = _selection_args([0], 3)
    assert tracker.contribution_wrt_selection(selected, n_selected) is sep_selected


def test_add_remove_updates_contribution(tracker: SeparationTracker):
    # --- arrange -----------------------------------------
    selected, n_selected = _selection_args([0, 2], 5)

    # --- act ---------------------------------------------
    tracker.add(np.int32(0))
    tracker.add(np.int32(2))

    # --- assert ------------------------------------------
    # points [0, 1, 3, 6, 10]; selection {0 (=0.0), 2 (=3.0)}: distance of each point to nearest selected
    np.testing.assert_allclose(tracker.contribution_wrt_selection(selected, n_selected), [3, 1, 3, 3, 7])

    # --- act (remove) ------------------------------------
    tracker.remove(np.int32(2), new_selection=np.array([0], dtype=np.int32))

    # --- assert ------------------------------------------
    selected, n_selected = _selection_args([0], 5)
    np.testing.assert_allclose(tracker.contribution_wrt_selection(selected, n_selected), [np.inf, 1, 3, 6, 10])


def test_add_many_remove_many_match_singles(tracker: SeparationTracker):
    # --- arrange -----------------------------------------
    other = tracker.copy()

    # --- act ---------------------------------------------
    tracker.add_many(np.array([1, 3, 4], dtype=np.int32))
    other.add(np.int32(1))
    other.add(np.int32(3))
    other.add(np.int32(4))

    tracker.remove_many(np.array([3, 4], dtype=np.int32), new_selection=np.array([1], dtype=np.int32))
    other.remove(np.int32(3), new_selection=np.array([1], dtype=np.int32))
    other.remove(np.int32(4), new_selection=np.array([1], dtype=np.int32))

    # --- assert ------------------------------------------
    selected, n_selected = _selection_args([1], 5)
    np.testing.assert_array_equal(
        tracker.contribution_wrt_selection(selected, n_selected),
        other.contribution_wrt_selection(selected, n_selected),
    )


def test_copy_is_independent(tracker: SeparationTracker):
    # --- arrange -----------------------------------------
    tracker.add(np.int32(0))
    clone = tracker.copy()

    # --- act ---------------------------------------------
    tracker.add(np.int32(4))

    # --- assert ------------------------------------------
    selected, n_selected = _selection_args([0], 5)
    np.testing.assert_allclose(clone.contribution_wrt_selection(selected, n_selected), [np.inf, 1, 3, 6, 10])
    # the immutable store and global contributions are shared by contract, not duplicated
    assert clone._store is tracker._store
    assert clone.contribution_wrt_dataset is tracker.contribution_wrt_dataset


def test_snapshot_stack(tracker: SeparationTracker):
    # --- arrange -----------------------------------------
    tracker.add(np.int32(0))

    # --- act & assert ------------------------------------
    # two nested snapshots: pop the inner one restoring, the outer one keeping
    tracker.push_snapshot()
    tracker.add(np.int32(3))
    selected_inner, n_selected_inner = _selection_args([0, 3], 5)
    contribution_inner = tracker.contribution_wrt_selection(selected_inner, n_selected_inner).copy()

    tracker.push_snapshot()
    tracker.add(np.int32(4))
    tracker.pop_snapshot(restore=True)  # restore to the inner snapshot ({0, 3})
    np.testing.assert_array_equal(
        tracker.contribution_wrt_selection(selected_inner, n_selected_inner), contribution_inner
    )

    tracker.pop_snapshot(restore=False)  # keep {0, 3}; the outer snapshot is only discarded
    np.testing.assert_array_equal(
        tracker.contribution_wrt_selection(selected_inner, n_selected_inner), contribution_inner
    )


# =================================================================================================
#  Kernels
# =================================================================================================
def test_compute_separation():
    """Check if compute_separation produces correct separation values."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)
    m = vectors.shape[0]
    d_squared = squareform(d)

    expected_separation = np.full(m, fill_value=np.inf, dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i != j:
                dist = d_squared[i, j]
                if dist < expected_separation[i]:
                    expected_separation[i] = dist

    # --- act ---------------------------------------------
    separation = compute_separation(DistanceStore.condensed(d, n=m))

    # --- assert ------------------------------------------
    np.testing.assert_allclose(separation, expected_separation)


def test_update_separation_add():
    """Check if update_separation_add correctly updates separation after adding a vector."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2], [1.1, 0]], dtype=np.float32)
    m = vectors.shape[0]
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)
    d_squared = squareform(d)

    # initial separation, assuming vector 0 forms the initial selection
    separation = np.array(
        [
            np.inf,
            d_squared[1, 0],
            d_squared[2, 0],
            d_squared[3, 0],
            d_squared[4, 0],
        ],
        dtype=np.float32,
    )

    i_added = 2  # Adding the vector [1, 0]

    expected_separation = np.array(
        [
            d_squared[0, 2],
            min(d_squared[1, 0], d_squared[1, 2]),
            d_squared[2, 0],
            min(d_squared[3, 0], d_squared[3, 2]),
            min(d_squared[4, 0], d_squared[4, 2]),
        ],
        dtype=np.float32,
    )

    # --- act ---------------------------------------------
    update_separation_add(separation, DistanceStore.condensed(d, n=m), np.int32(i_added))

    # --- assert ------------------------------------------
    np.testing.assert_allclose(separation, expected_separation)


def test_update_separation_remove():
    """Check if update_separation_remove correctly updates separation after adding a vector."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2], [1.1, 0]], dtype=np.float32)
    m = vectors.shape[0]
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)
    d_squared = squareform(d)

    # initial separation, assuming vector 0 & 2 form the initial selection
    separation = np.array(
        [
            d_squared[0, 2],
            min(d_squared[1, 0], d_squared[1, 2]),
            d_squared[2, 0],
            min(d_squared[3, 0], d_squared[3, 2]),
            min(d_squared[4, 0], d_squared[4, 2]),
        ],
        dtype=np.float32,
    )

    i_removed = 2  # Removing the vector [1, 0]

    expected_separation = np.array(
        [
            np.inf,
            d_squared[1, 0],
            d_squared[2, 0],
            d_squared[3, 0],
            d_squared[4, 0],
        ],
        dtype=np.float32,
    )

    # --- act ---------------------------------------------
    update_separation_remove(
        separation, DistanceStore.condensed(d, n=m), np.int32(i_removed), np.array([0], dtype=np.int32)
    )

    # --- assert ------------------------------------------
    np.testing.assert_allclose(separation, expected_separation)
