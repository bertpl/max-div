import numpy as np
import pytest
from numpy import random
from scipy.spatial.distance import squareform

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import compute_pdist
from max_div._core.solver._signals import MeanDistanceTracker

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 20


@pytest.fixture
def pdist() -> np.ndarray:
    rng = random.default_rng(seed=20260713)
    vectors = rng.random((N, 3)).astype(np.float32)
    return compute_pdist(vectors, DistanceMetric.L2_EUCLIDEAN)


@pytest.fixture
def tracker(pdist: np.ndarray) -> MeanDistanceTracker:
    return MeanDistanceTracker(pdist, np.int32(N))


def _selection_args(indices: list[int]) -> tuple[np.ndarray, np.int32]:
    """Build the (selected, n_selected) argument pair for signal reads from a list of selected indices."""
    selected = np.full(N, False, dtype=np.bool)
    selected[indices] = True
    return selected, np.int32(len(indices))


def _brute_force_signal(pdist: np.ndarray, indices: list[int]) -> np.ndarray:
    """Compute the mean-distance signal from scratch: mean distance of each point to its selected neighbors."""
    d_squared = squareform(pdist).astype(np.float64)
    selected = np.full(N, False, dtype=np.bool)
    selected[indices] = True
    sums = d_squared[:, selected].sum(axis=1) if indices else np.zeros(N, dtype=np.float64)
    divisor = np.maximum(len(indices) - selected, 1)
    return (sums / divisor).astype(np.float32)


# =================================================================================================
#  Tests
# =================================================================================================
def test_construction_fresh(tracker: MeanDistanceTracker, pdist: np.ndarray):
    # --- arrange -----------------------------------------
    selected, n_selected = _selection_args([])
    expected_global = (squareform(pdist).astype(np.float64).sum(axis=1) / (N - 1)).astype(np.float32)

    # --- assert ------------------------------------------
    assert tracker.global_signal.dtype == np.float32
    np.testing.assert_allclose(tracker.global_signal, expected_global, rtol=1e-6)
    # empty selection: all signals 0.0 (no selected neighbors)
    np.testing.assert_array_equal(tracker.full_signal(selected, n_selected), np.zeros(N, dtype=np.float32))


def test_construction_precomputed_arrays_skip_recompute(pdist: np.ndarray):
    # --- arrange -----------------------------------------
    global_signal = np.arange(N, dtype=np.float32)
    dist_sums = np.arange(N, dtype=np.float64)

    # --- act ---------------------------------------------
    tracker = MeanDistanceTracker(pdist, np.int32(N), global_signal=global_signal, dist_sums=dist_sums)

    # --- assert ------------------------------------------
    assert tracker.global_signal is global_signal  # taken as-is, not recomputed
    assert tracker._dist_sums is dist_sums


def test_signal_matches_brute_force_incrementally(tracker: MeanDistanceTracker, pdist: np.ndarray):
    # --- arrange -----------------------------------------
    selection: list[int] = []

    # --- act / assert ------------------------------------
    for index in [3, 17, 0, 9, 12]:
        tracker.add(np.int32(index))
        selection.append(index)
        selected, n_selected = _selection_args(selection)
        np.testing.assert_allclose(
            tracker.full_signal(selected, n_selected), _brute_force_signal(pdist, selection), rtol=1e-6
        )

    for index in [0, 17]:
        tracker.remove(np.int32(index), new_selection=np.array([], dtype=np.int32))
        selection.remove(index)
        selected, n_selected = _selection_args(selection)
        np.testing.assert_allclose(
            tracker.full_signal(selected, n_selected), _brute_force_signal(pdist, selection), rtol=1e-6
        )


def test_membership_aware_divisor(tracker: MeanDistanceTracker, pdist: np.ndarray):
    """A selected point's mean divides by (n_selected - 1); a non-selected point's by n_selected."""

    # --- arrange -----------------------------------------
    d_squared = squareform(pdist).astype(np.float64)
    tracker.add(np.int32(2))
    tracker.add(np.int32(5))
    selected, n_selected = _selection_args([2, 5])

    # --- act ---------------------------------------------
    signal = tracker.full_signal(selected, n_selected)

    # --- assert ------------------------------------------
    # selected point 2: one real neighbor (5) -> mean = d(2,5) / 1
    assert signal[2] == pytest.approx(d_squared[2, 5], rel=1e-6)
    # non-selected point 0: two neighbors -> mean = (d(0,2) + d(0,5)) / 2
    assert signal[0] == pytest.approx((d_squared[0, 2] + d_squared[0, 5]) / 2, rel=1e-6)


def test_invariant_random_operations_match_recompute(tracker: MeanDistanceTracker, pdist: np.ndarray):
    """After arbitrary add/remove/snapshot sequences, signals must match a brute-force recompute."""

    # --- arrange -----------------------------------------
    rng = random.default_rng(seed=7)
    selection: list[int] = []
    snapshot_selection: list[int] | None = None

    # --- act / assert ------------------------------------
    for _ in range(200):
        can_restore = snapshot_selection is not None
        options = ["add", "remove", "set_snapshot"] + (["restore_snapshot"] if can_restore else [])
        match rng.choice(options):
            case "add" if len(selection) < N:
                index = int(rng.choice([i for i in range(N) if i not in selection]))
                tracker.add(np.int32(index))
                selection.append(index)
            case "remove" if selection:
                index = int(rng.choice(selection))
                selection.remove(index)
                tracker.remove(np.int32(index), new_selection=np.array(selection, dtype=np.int32))
            case "set_snapshot":
                tracker.set_snapshot()
                snapshot_selection = selection.copy()
            case "restore_snapshot":
                tracker.restore_snapshot()
                selection = snapshot_selection.copy()  # ty: ignore[possibly-unbound-attribute]  # gated by can_restore
                snapshot_selection = None

        selected, n_selected = _selection_args(selection)
        np.testing.assert_allclose(
            tracker.full_signal(selected, n_selected), _brute_force_signal(pdist, selection), rtol=1e-5
        )


def test_copy_is_independent(tracker: MeanDistanceTracker):
    # --- arrange -----------------------------------------
    tracker.add(np.int32(0))
    clone = tracker.copy()
    selected, n_selected = _selection_args([0])
    signal_before = clone.full_signal(selected, n_selected).copy()

    # --- act ---------------------------------------------
    tracker.add(np.int32(4))

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(clone.full_signal(selected, n_selected), signal_before)
    assert clone.global_signal is not tracker.global_signal
