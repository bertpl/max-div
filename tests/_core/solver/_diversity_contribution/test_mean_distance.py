import numpy as np
import pytest
from numpy import random
from scipy.spatial.distance import squareform

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._diversity_contribution import MeanDistanceTracker
from max_div._core.solver._diversity_contribution._mean_distance import (
    backend_for,
)
from tests._core.metrics._distance.helpers import condensed_distances

from .helpers import selection_args

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 20


@pytest.fixture
def pdist() -> np.ndarray:
    rng = random.default_rng(seed=20260713)
    vectors = rng.random((N, 3)).astype(np.float32)
    return condensed_distances(vectors, DistanceMetric.l2_euclidean())


@pytest.fixture
def tracker(pdist: np.ndarray) -> MeanDistanceTracker:
    return MeanDistanceTracker(DistanceStore.full_matrix(squareform(pdist)))


def _brute_force_contribution(pdist: np.ndarray, indices: list[int]) -> np.ndarray:
    """Compute the mean-distance contribution from scratch: mean distance of each point to its selected neighbors."""
    d_squared = squareform(pdist).astype(np.float64)
    selected = np.full(N, False, dtype=np.bool)
    selected[indices] = True
    sums = d_squared[:, selected].sum(axis=1) if indices else np.zeros(N, dtype=np.float64)
    divisor = np.maximum(len(indices) - selected, 1)
    return (sums / divisor).astype(np.float32)


# =================================================================================================
#  Tests
# =================================================================================================
def test_construction_fresh(tracker: MeanDistanceTracker):
    # --- arrange ----------------------
    selected, n_selected = selection_args([], N)

    # --- assert -----------------------
    assert tracker.store.n == N
    # empty selection: all contributions 0.0 (no selected neighbors)
    np.testing.assert_array_equal(
        tracker.contribution_wrt_selection(selected, n_selected), np.zeros(N, dtype=np.float32)
    )


def test_contribution_matches_brute_force_incrementally(tracker: MeanDistanceTracker, pdist: np.ndarray):
    # --- arrange ----------------------
    selection: list[int] = []

    # --- act / assert -----------------
    for index in [3, 17, 0, 9, 12]:
        tracker.add(np.int32(index))
        selection.append(index)
        selected, n_selected = selection_args(selection, N)
        np.testing.assert_allclose(
            tracker.contribution_wrt_selection(selected, n_selected),
            _brute_force_contribution(pdist, selection),
            rtol=1e-6,
        )

    for index in [0, 17]:
        tracker.remove(np.int32(index), new_selection=np.array([], dtype=np.int32))
        selection.remove(index)
        selected, n_selected = selection_args(selection, N)
        np.testing.assert_allclose(
            tracker.contribution_wrt_selection(selected, n_selected),
            _brute_force_contribution(pdist, selection),
            rtol=1e-6,
        )


def test_membership_aware_divisor(tracker: MeanDistanceTracker, pdist: np.ndarray):
    """A selected point's mean divides by (n_selected - 1); a non-selected point's by n_selected."""

    # --- arrange ----------------------
    d_squared = squareform(pdist).astype(np.float64)
    tracker.add(np.int32(2))
    tracker.add(np.int32(5))
    selected, n_selected = selection_args([2, 5], N)

    # --- act --------------------------
    contribution = tracker.contribution_wrt_selection(selected, n_selected)

    # --- assert -----------------------
    # selected point 2: one real neighbor (5) -> mean = d(2,5) / 1
    assert contribution[2] == pytest.approx(d_squared[2, 5], rel=1e-6)
    # non-selected point 0: two neighbors -> mean = (d(0,2) + d(0,5)) / 2
    assert contribution[0] == pytest.approx((d_squared[0, 2] + d_squared[0, 5]) / 2, rel=1e-6)


def test_invariant_random_operations_match_recompute(tracker: MeanDistanceTracker, pdist: np.ndarray):
    """After arbitrary add/remove/snapshot sequences, contributions must match a brute-force recompute."""

    # --- arrange ----------------------
    rng = random.default_rng(seed=7)
    selection: list[int] = []
    snapshot_selections: list[list[int]] = []  # mirrors the tracker's snapshot stack

    # --- act / assert -----------------
    for _ in range(200):
        options = ["add", "remove", "push_snapshot"] + (["pop_restore", "pop_keep"] if snapshot_selections else [])
        match rng.choice(options):
            case "add" if len(selection) < N:
                index = int(rng.choice([i for i in range(N) if i not in selection]))
                tracker.add(np.int32(index))
                selection.append(index)
            case "remove" if selection:
                index = int(rng.choice(selection))
                selection.remove(index)
                tracker.remove(np.int32(index), new_selection=np.array(selection, dtype=np.int32))
            case "push_snapshot":
                tracker.push_snapshot()
                snapshot_selections.append(selection.copy())
            case "pop_restore":
                tracker.pop_snapshot(restore=True)
                selection = snapshot_selections.pop()
            case "pop_keep":
                tracker.pop_snapshot(restore=False)
                snapshot_selections.pop()

        selected, n_selected = selection_args(selection, N)
        np.testing.assert_allclose(
            tracker.contribution_wrt_selection(selected, n_selected),
            _brute_force_contribution(pdist, selection),
            rtol=1e-5,
        )


# =================================================================================================
#  Kernels
# =================================================================================================
def test_update_distance_sums_add_remove():
    """Incremental add/remove updates match brute-force sums over the selection at every step."""

    # --- arrange ----------------------
    rng = np.random.default_rng(20260713)
    vectors = rng.standard_normal((20, 3)).astype(np.float32)
    m = vectors.shape[0]
    d = condensed_distances(vectors, metric=DistanceMetric.l2_euclidean())
    d_squared = squareform(d).astype(np.float64)

    dist_sums = np.zeros(m, dtype=np.float64)
    selection: list[int] = []

    def expected_sums() -> np.ndarray:
        # brute-force: each point's sum of distances to the selected points (self-distance is 0)
        return d_squared[:, selection].sum(axis=1) if selection else np.zeros(m, dtype=np.float64)

    # --- act / assert -----------------
    for index in [3, 17, 0, 9, 12]:
        store = DistanceStore.full_matrix(squareform(d))
        backend_for(store).add(dist_sums, store, np.int32(index))
        selection.append(index)
        np.testing.assert_allclose(dist_sums, expected_sums(), rtol=1e-6)

    for index in [0, 3, 12]:
        store = DistanceStore.full_matrix(squareform(d))
        backend_for(store).remove(dist_sums, store, np.int32(index))
        selection.remove(index)
        np.testing.assert_allclose(dist_sums, expected_sums(), rtol=1e-6)


def test_update_distance_sums_own_entry_untouched():
    """A point's own entry is unchanged by adding/removing that point (its self-distance is 0)."""

    # --- arrange ----------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    m = vectors.shape[0]
    d = condensed_distances(vectors, metric=DistanceMetric.l2_euclidean())
    dist_sums = np.zeros(m, dtype=np.float64)

    # selection {1}: point 1's own entry stays 0 (no other selected points yet)
    store = DistanceStore.full_matrix(squareform(d))
    backend_for(store).add(dist_sums, store, np.int32(1))
    assert dist_sums[1] == 0.0

    # --- act --------------------------
    # selection {1, 2}: point 2's own entry must equal its distance to point 1 only
    backend_for(store).add(dist_sums, store, np.int32(2))

    # --- assert -----------------------
    assert dist_sums[2] == pytest.approx(squareform(d)[2, 1])


# =================================================================================================
#  Backend equivalence
# =================================================================================================
# One backend module per storage layout means the same logic exists once per layout, so a fix
# applied to only some of them would pass review looking complete.  Driving every backend through the same
# operations against a brute-force recompute is the guard against that.
@pytest.mark.parametrize("backend", ["full_matrix", "lazy"])
def test_backend_matches_brute_force_over_random_operations(backend: str):
    """Random add/remove sequences must match a brute-force recompute, on every layout."""

    # --- arrange ----------------------
    rng = random.default_rng(20260805)
    vectors = rng.random((N, 3)).astype(np.float32)
    condensed = condensed_distances(vectors, DistanceMetric.l2_euclidean())
    store = {
        "full_matrix": DistanceStore.full_matrix_from_vectors(vectors, DistanceMetric.l2_euclidean()),
        "lazy": DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean()),
    }[backend]
    tracker = MeanDistanceTracker(store)
    selection: list[int] = []

    # --- act / assert -----------------
    for _ in range(120):
        must_remove = len(selection) == N  # nothing left to add once everything is selected
        if selection and (must_remove or rng.random() < 0.4):
            index = int(rng.choice(selection))
            selection.remove(index)
            tracker.remove(np.int32(index), new_selection=np.array(selection, dtype=np.int32))
        else:
            index = int(rng.choice([i for i in range(N) if i not in selection]))
            tracker.add(np.int32(index))
            selection.append(index)

        selected, n_selected = selection_args(selection, N)
        np.testing.assert_allclose(
            tracker.contribution_wrt_selection(selected, n_selected),
            _brute_force_contribution(condensed, selection),
            rtol=1e-5,
            err_msg=f"{backend} diverged",
        )


def test_reset_returns_to_empty_selection(tracker: MeanDistanceTracker):
    """Reset returns distance sums to the empty-selection zeros."""
    # --- arrange ----------------------
    tracker.add(np.int32(0))
    tracker.add(np.int32(2))
    selected = np.full(N, False, dtype=np.bool)

    # --- act --------------------------
    tracker.reset()

    # --- assert -----------------------
    assert np.all(tracker.contribution_wrt_selection(selected, np.int32(0)) == 0.0)


@pytest.mark.parametrize("backend", ["full_matrix", "lazy"])
def test_remove_trial_matches_remove_on_the_selected_entries(backend: str):
    """The selected-only subtraction agrees with the full update wherever the score reads."""
    # --- arrange ----------------------
    rng = random.default_rng(20260902)
    vectors = rng.random((N, 3)).astype(np.float32)
    metric = DistanceMetric.l2_euclidean()
    store = {
        "full_matrix": DistanceStore.full_matrix_from_vectors(vectors, metric),
        "lazy": DistanceStore.lazy(vectors, metric),
    }[backend]
    indices = [1, 4, 7, 12, 18]
    full, trial = MeanDistanceTracker(store), MeanDistanceTracker(store)
    for i in indices:
        full.add(np.int32(i))
        trial.add(np.int32(i))
    new_selection = np.array([1, 4, 12, 18], dtype=np.int32)

    # --- act --------------------------
    full.remove(np.int32(7), new_selection)
    trial.remove_trial(np.int32(7), new_selection)

    # --- assert -----------------------
    selected, n_selected = selection_args([1, 4, 12, 18], N)
    np.testing.assert_array_equal(
        trial.contribution_wrt_selection(selected, n_selected)[new_selection],
        full.contribution_wrt_selection(selected, n_selected)[new_selection],
    )
