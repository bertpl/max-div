import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import compute_pdist, compute_separation
from max_div._core.solver._signals import SeparationTracker


# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
@pytest.fixture
def tracker() -> SeparationTracker:
    vectors = np.array([[0.0], [1.0], [3.0], [6.0], [10.0]], dtype=np.float32)
    pdist = compute_pdist(vectors, DistanceMetric.L1_MANHATTAN)
    return SeparationTracker(pdist, np.int32(vectors.shape[0]))


def _selection_args(indices: list[int], n: int) -> tuple[np.ndarray, np.int32]:
    """Build the (selected, n_selected) argument pair for signal reads from a list of selected indices."""
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
    # global signal: nearest-neighbor distances of points [0, 1, 3, 6, 10] on a line
    np.testing.assert_allclose(tracker.global_signal, [1, 1, 2, 3, 4])
    # empty selection: all signals +inf
    assert np.all(np.isinf(tracker.full_signal(selected, n_selected)))


def test_construction_precomputed_arrays_skip_recompute():
    # --- arrange -----------------------------------------
    vectors = np.array([[0.0], [1.0], [5.0]], dtype=np.float32)
    pdist = compute_pdist(vectors, DistanceMetric.L1_MANHATTAN)
    sep_global = compute_separation(pdist, np.int32(3))
    sep_selected = np.array([7.0, 8.0, 9.0], dtype=np.float32)

    # --- act ---------------------------------------------
    tracker = SeparationTracker(pdist, np.int32(3), sep_global=sep_global, sep_selected=sep_selected)

    # --- assert ------------------------------------------
    assert tracker.global_signal is sep_global  # taken as-is, not recomputed
    selected, n_selected = _selection_args([0], 3)
    assert tracker.full_signal(selected, n_selected) is sep_selected


def test_add_remove_updates_signal(tracker: SeparationTracker):
    # --- arrange -----------------------------------------
    selected, n_selected = _selection_args([0, 2], 5)

    # --- act ---------------------------------------------
    tracker.add(np.int32(0))
    tracker.add(np.int32(2))

    # --- assert ------------------------------------------
    # points [0, 1, 3, 6, 10]; selection {0 (=0.0), 2 (=3.0)}: distance of each point to nearest selected
    np.testing.assert_allclose(tracker.full_signal(selected, n_selected), [3, 1, 3, 3, 7])

    # --- act (remove) ------------------------------------
    tracker.remove(np.int32(2), new_selection=np.array([0], dtype=np.int32))

    # --- assert ------------------------------------------
    selected, n_selected = _selection_args([0], 5)
    np.testing.assert_allclose(tracker.full_signal(selected, n_selected), [np.inf, 1, 3, 6, 10])


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
        tracker.full_signal(selected, n_selected),
        other.full_signal(selected, n_selected),
    )


def test_copy_is_independent(tracker: SeparationTracker):
    # --- arrange -----------------------------------------
    tracker.add(np.int32(0))
    clone = tracker.copy()

    # --- act ---------------------------------------------
    tracker.add(np.int32(4))

    # --- assert ------------------------------------------
    selected, n_selected = _selection_args([0], 5)
    np.testing.assert_allclose(clone.full_signal(selected, n_selected), [np.inf, 1, 3, 6, 10])
    assert clone.global_signal is not tracker.global_signal


def test_snapshot_restore(tracker: SeparationTracker):
    # --- arrange -----------------------------------------
    tracker.add(np.int32(0))
    selected, n_selected = _selection_args([0], 5)
    signal_before = tracker.full_signal(selected, n_selected).copy()

    # --- act ---------------------------------------------
    tracker.set_snapshot()
    tracker.add(np.int32(3))
    tracker.restore_snapshot()

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(tracker.full_signal(selected, n_selected), signal_before)
