import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric, DiversityContributionFamily, DiversityMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.solver._diversity_contribution import (
    DiversityContributionTrackers,
    MeanDistanceTracker,
    SeparationTracker,
    selected_contributions_slot,
)

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 6


@pytest.fixture
def store() -> DistanceStore:
    vectors = np.array([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0]], dtype=np.float32)
    return DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=N)


# =================================================================================================
#  Tests
# =================================================================================================
def test_for_metrics_single_family(store: DistanceStore):
    # --- act --------------------------
    trackers = DiversityContributionTrackers.for_metrics(
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[DiversityMetric.NON_ZERO_SEPARATION_FRAC],
        store=store,
    )

    # --- assert -----------------------
    # both metrics are separation-family -> exactly one tracker, which is also the main one
    assert len(trackers._trackers) == 1
    assert type(trackers.main) is SeparationTracker
    assert trackers.main is trackers._trackers[0]


def test_for_metrics_mixed_families(store: DistanceStore):
    # --- act --------------------------
    trackers = DiversityContributionTrackers.for_metrics(
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[DiversityMetric.MIN_SEPARATION, DiversityMetric.MEAN_SEPARATION],
        store=store,
    )

    # --- assert -----------------------
    # duplicate families are deduplicated
    assert len(trackers._trackers) == 1
    assert type(trackers.main) is SeparationTracker


def test_mutations_fan_out_to_all_trackers(store: DistanceStore):
    # --- arrange ----------------------
    # hand-built two-family set, mirroring what a mixed-metric configuration would construct
    sep, mean = SeparationTracker(store), MeanDistanceTracker(store)
    trackers = DiversityContributionTrackers(
        trackers_by_family={
            DiversityContributionFamily.SEPARATION: sep,
            DiversityContributionFamily.MEAN_DISTANCE: mean,
        },
        main_family=DiversityContributionFamily.MEAN_DISTANCE,
    )
    sep_ref, mean_ref = SeparationTracker(store), MeanDistanceTracker(store)
    selected = np.full(N, False, dtype=np.bool)
    selected[[0, 3]] = True

    # --- act --------------------------
    trackers.add(np.int32(0))
    trackers.add_many(np.array([2, 3], dtype=np.int32))
    trackers.push_snapshot()
    trackers.remove_many(np.array([2, 3], dtype=np.int32), new_selection=np.array([0], dtype=np.int32))
    trackers.pop_snapshot(restore=True)
    trackers.remove(np.int32(2), new_selection=np.array([0, 3], dtype=np.int32))

    for ref in (sep_ref, mean_ref):
        ref.add(np.int32(0))
        ref.add_many(np.array([2, 3], dtype=np.int32))
        ref.remove(np.int32(2), new_selection=np.array([0, 3], dtype=np.int32))

    # --- assert -----------------------
    assert trackers.main is mean
    np.testing.assert_array_equal(
        sep.contribution_wrt_selection(selected, np.int32(2)),
        sep_ref.contribution_wrt_selection(selected, np.int32(2)),
    )
    np.testing.assert_array_equal(
        mean.contribution_wrt_selection(selected, np.int32(2)),
        mean_ref.contribution_wrt_selection(selected, np.int32(2)),
    )


def test_copy_is_independent(store: DistanceStore):
    # --- arrange ----------------------
    trackers = DiversityContributionTrackers.for_metrics(
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[],
        store=store,
    )
    trackers.add(np.int32(0))
    clone = trackers.copy()
    selected = np.full(N, False, dtype=np.bool)
    selected[0] = True
    before = clone.main.contribution_wrt_selection(selected, np.int32(1)).copy()

    # --- act --------------------------
    trackers.add(np.int32(4))

    # --- assert -----------------------
    assert clone.main is not trackers.main
    np.testing.assert_array_equal(clone.main.contribution_wrt_selection(selected, np.int32(1)), before)


def test_selected_contributions_slots(store: DistanceStore):
    """Active families fill their slot with the selected vectors' values; untracked slots are empty."""

    # --- arrange ----------------------
    trackers = DiversityContributionTrackers.for_metrics(
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[],
        store=store,
    )
    trackers.add(np.int32(0))
    trackers.add(np.int32(2))  # selection: points 0.0 and 3.0 on a line
    selected = np.full(N, False, dtype=np.bool)
    selected[[0, 2]] = True

    # --- act --------------------------
    contributions = trackers.selected_contributions(selected, np.int32(2))

    # --- assert -----------------------
    sep_slot = selected_contributions_slot(DiversityContributionFamily.SEPARATION)
    mean_slot = selected_contributions_slot(DiversityContributionFamily.MEAN_DISTANCE)
    assert sep_slot != mean_slot
    np.testing.assert_allclose(contributions[sep_slot], [3.0, 3.0])  # separation of the two selected points
    assert contributions[mean_slot].size == 0  # mean-distance family untracked -> shared empty slot
