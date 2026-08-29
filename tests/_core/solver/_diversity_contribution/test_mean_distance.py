import numpy as np
import pytest
from numpy import random
from scipy.spatial.distance import squareform

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.solver._diversity_contribution import MeanDistanceTracker
from max_div._core.solver._diversity_contribution._mean_distance import (
    backend_for,
)

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 20


@pytest.fixture
def pdist() -> np.ndarray:
    rng = random.default_rng(seed=20260713)
    vectors = rng.random((N, 3)).astype(np.float32)
    return compute_pdist(vectors, DistanceMetric.l2_euclidean())


@pytest.fixture
def tracker(pdist: np.ndarray) -> MeanDistanceTracker:
    return MeanDistanceTracker(DistanceStore.condensed(pdist, n=N))


def _selection_args(indices: list[int]) -> tuple[np.ndarray, np.int32]:
    """Build the (selected, n_selected) argument pair for contribution reads from a list of selected indices."""
    selected = np.full(N, False, dtype=np.bool)
    selected[indices] = True
    return selected, np.int32(len(indices))


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
def test_construction_fresh(tracker: MeanDistanceTracker, pdist: np.ndarray):
    # --- arrange ----------------------
    selected, n_selected = _selection_args([])
    expected_global = (squareform(pdist).astype(np.float64).sum(axis=1) / (N - 1)).astype(np.float32)

    # --- assert -----------------------
    assert tracker.contribution_wrt_dataset.dtype == np.float32
    np.testing.assert_allclose(tracker.contribution_wrt_dataset, expected_global, rtol=1e-6)
    # empty selection: all contributions 0.0 (no selected neighbors)
    np.testing.assert_array_equal(
        tracker.contribution_wrt_selection(selected, n_selected), np.zeros(N, dtype=np.float32)
    )


def test_construction_precomputed_arrays_skip_recompute(pdist: np.ndarray):
    # --- arrange ----------------------
    contribution_wrt_dataset = np.arange(N, dtype=np.float32)
    dist_sums = np.arange(N, dtype=np.float64)

    # --- act --------------------------
    tracker = MeanDistanceTracker(
        DistanceStore.condensed(pdist, n=N), contribution_wrt_dataset=contribution_wrt_dataset, dist_sums=dist_sums
    )

    # --- assert -----------------------
    assert tracker.contribution_wrt_dataset is contribution_wrt_dataset  # taken as-is, not recomputed
    assert tracker._dist_sums is dist_sums


def test_contribution_matches_brute_force_incrementally(tracker: MeanDistanceTracker, pdist: np.ndarray):
    # --- arrange ----------------------
    selection: list[int] = []

    # --- act / assert -----------------
    for index in [3, 17, 0, 9, 12]:
        tracker.add(np.int32(index))
        selection.append(index)
        selected, n_selected = _selection_args(selection)
        np.testing.assert_allclose(
            tracker.contribution_wrt_selection(selected, n_selected),
            _brute_force_contribution(pdist, selection),
            rtol=1e-6,
        )

    for index in [0, 17]:
        tracker.remove(np.int32(index), new_selection=np.array([], dtype=np.int32))
        selection.remove(index)
        selected, n_selected = _selection_args(selection)
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
    selected, n_selected = _selection_args([2, 5])

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

        selected, n_selected = _selection_args(selection)
        np.testing.assert_allclose(
            tracker.contribution_wrt_selection(selected, n_selected),
            _brute_force_contribution(pdist, selection),
            rtol=1e-5,
        )


def test_lazy_global_targeted_read_computes_only_requested(tracker: MeanDistanceTracker, pdist: np.ndarray):
    # --- arrange ----------------------
    expected_global = (squareform(pdist).astype(np.float64).sum(axis=1) / (N - 1)).astype(np.float32)
    requested = np.array([2, 11, 5], dtype=np.int32)

    # --- act --------------------------
    values = tracker.contribution_wrt_dataset_for(requested)

    # --- assert -----------------------
    np.testing.assert_allclose(values, expected_global[requested], rtol=1e-6)
    # only the requested elements are computed; the rest of the cache is still pending
    untouched = np.setdiff1d(np.arange(N), requested)
    assert np.all(np.isnan(tracker._contribution_wrt_dataset[untouched]))
    # the returned array is a fresh copy, not a view into the cache
    values[0] = -1.0
    assert tracker._contribution_wrt_dataset[2] != -1.0


def test_lazy_global_cache_shared_across_copies(tracker: MeanDistanceTracker):
    # --- arrange ----------------------
    clone = tracker.copy()

    # --- act --------------------------
    clone.contribution_wrt_dataset_for(np.array([4], dtype=np.int32))

    # --- assert -----------------------
    # an element computed through the clone is visible through the original (one shared cache)
    assert not np.isnan(tracker._contribution_wrt_dataset[4])


def test_copy_is_independent(tracker: MeanDistanceTracker):
    # --- arrange ----------------------
    tracker.add(np.int32(0))
    clone = tracker.copy()
    selected, n_selected = _selection_args([0])
    contribution_before = clone.contribution_wrt_selection(selected, n_selected).copy()

    # --- act --------------------------
    tracker.add(np.int32(4))

    # --- assert -----------------------
    np.testing.assert_array_equal(clone.contribution_wrt_selection(selected, n_selected), contribution_before)
    # the immutable store and global contributions are shared by contract, not duplicated
    assert clone._store is tracker._store
    assert clone.contribution_wrt_dataset is tracker.contribution_wrt_dataset


# =================================================================================================
#  Kernels
# =================================================================================================
def test_compute_mean_distance_elements_partial_fill():
    """The elements kernel fills exactly the requested elements, with brute-force mean values."""

    # --- arrange ----------------------
    rng = np.random.default_rng(20260713)
    vectors = rng.standard_normal((30, 4)).astype(np.float32)
    m = vectors.shape[0]
    d = compute_pdist(vectors, metric=DistanceMetric.l2_euclidean())
    expected = (squareform(d).astype(np.float64).sum(axis=1) / (m - 1)).astype(np.float32)
    out = np.full(m, np.nan, dtype=np.float32)
    requested = np.array([0, 7, 29, 13], dtype=np.int32)

    # --- act --------------------------
    store = DistanceStore.condensed(d, n=m)
    backend_for(store).elements(out, store, requested)

    # --- assert -----------------------
    np.testing.assert_allclose(out[requested], expected[requested], rtol=1e-6)
    untouched = np.setdiff1d(np.arange(m), requested)
    assert np.all(np.isnan(out[untouched]))  # only the requested elements were written


def test_update_distance_sums_add_remove():
    """Incremental add/remove updates match brute-force sums over the selection at every step."""

    # --- arrange ----------------------
    rng = np.random.default_rng(20260713)
    vectors = rng.standard_normal((20, 3)).astype(np.float32)
    m = vectors.shape[0]
    d = compute_pdist(vectors, metric=DistanceMetric.l2_euclidean())
    d_squared = squareform(d).astype(np.float64)

    dist_sums = np.zeros(m, dtype=np.float64)
    selection: list[int] = []

    def expected_sums() -> np.ndarray:
        # brute-force: each point's sum of distances to the selected points (self-distance is 0)
        return d_squared[:, selection].sum(axis=1) if selection else np.zeros(m, dtype=np.float64)

    # --- act / assert -----------------
    for index in [3, 17, 0, 9, 12]:
        store = DistanceStore.condensed(d, n=m)
        backend_for(store).add(dist_sums, store, np.int32(index))
        selection.append(index)
        np.testing.assert_allclose(dist_sums, expected_sums(), rtol=1e-6)

    for index in [0, 3, 12]:
        store = DistanceStore.condensed(d, n=m)
        backend_for(store).remove(dist_sums, store, np.int32(index))
        selection.remove(index)
        np.testing.assert_allclose(dist_sums, expected_sums(), rtol=1e-6)


def test_update_distance_sums_own_entry_untouched():
    """A point's own entry is unchanged by adding/removing that point (its self-distance is 0)."""

    # --- arrange ----------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    m = vectors.shape[0]
    d = compute_pdist(vectors, metric=DistanceMetric.l2_euclidean())
    dist_sums = np.zeros(m, dtype=np.float64)

    # selection {1}: point 1's own entry stays 0 (no other selected points yet)
    store = DistanceStore.condensed(d, n=m)
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
# One backend module per storage layout means the same logic exists three times, so a fix applied
# to two of them would pass review looking complete.  Driving every backend through the same
# operations against a brute-force recompute is the guard against that.
@pytest.mark.parametrize("backend", ["full_matrix", "condensed", "lazy"])
def test_backend_matches_brute_force_over_random_operations(backend: str):
    """Random add/remove sequences must match a brute-force recompute, on every layout."""

    # --- arrange ----------------------
    rng = random.default_rng(20260805)
    vectors = rng.random((N, 3)).astype(np.float32)
    condensed = compute_pdist(vectors, DistanceMetric.l2_euclidean())
    store = {
        "full_matrix": DistanceStore.full_matrix_from_vectors(vectors, DistanceMetric.l2_euclidean()),
        "condensed": DistanceStore.condensed(condensed, n=N),
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

        selected, n_selected = _selection_args(selection)
        np.testing.assert_allclose(
            tracker.contribution_wrt_selection(selected, n_selected),
            _brute_force_contribution(condensed, selection),
            rtol=1e-5,
            err_msg=f"{backend} diverged",
        )


def test_reset_returns_to_empty_selection(tracker: MeanDistanceTracker):
    """Reset returns distance sums to the empty-selection zeros; the dataset-wide cache stays untouched."""
    # --- arrange ----------------------
    tracker.add(np.int32(0))
    tracker.add(np.int32(2))
    global_before = tracker.contribution_wrt_dataset.copy()
    selected = np.full(N, False, dtype=np.bool)

    # --- act --------------------------
    tracker.reset()

    # --- assert -----------------------
    assert np.all(tracker.contribution_wrt_selection(selected, np.int32(0)) == 0.0)
    np.testing.assert_array_equal(tracker.contribution_wrt_dataset, global_before)  # cache untouched
