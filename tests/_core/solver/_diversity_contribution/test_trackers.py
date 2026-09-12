import numpy as np
import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjectiveSimple,
)
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._diversity_contribution import (
    MAX_CONTRIBUTION_SLOTS,
    DiversityContributionTrackers,
    MeanDistanceTracker,
    SeparationTracker,
    build_diversity_contribution_slots,
)
from tests._core.solver.objectives import single_term_objective, tie_breaker_objectives

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 6
_VECTORS = np.array([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0]], dtype=np.float32)


@pytest.fixture
def store() -> DistanceStore:
    return DistanceStore.full_matrix_from_vectors(_VECTORS, DistanceMetric.l1_manhattan())


# =================================================================================================
#  build_diversity_contribution_slots
# =================================================================================================
@pytest.mark.parametrize(
    "diversity_objective, diversity_tie_breakers, expected_slots",
    [
        pytest.param(
            single_term_objective(DiversityMetric.MIN_SEPARATION),
            tie_breaker_objectives([DiversityMetric.MEAN_SEPARATION, DiversityMetric.MEAN_PAIRWISE_DISTANCE]),
            # MIN_SEPARATION and MEAN_SEPARATION share the separation family, so they share slot 0
            {
                (None, DiversityContributionFamily.SEPARATION): 0,
                (None, DiversityContributionFamily.MEAN_DISTANCE): 1,
            },
            id="first_seen_order_each_pair_once",
        ),
        pytest.param(
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, DistanceMetric.l2_euclidean()),
            [],
            {(DistanceMetric.l2_euclidean(), DiversityContributionFamily.SEPARATION): 0},
            id="pair_carries_the_distance",
        ),
    ],
)
def test_build_diversity_contribution_slots(
    diversity_objective: DiversityObjectiveSimple,
    diversity_tie_breakers: list,
    expected_slots: dict,
):
    """The main objective's pair is slot 0, each new pair follows once, and a pair keeps its own distance."""
    # --- act / assert -----------------
    assert build_diversity_contribution_slots(diversity_objective, diversity_tie_breakers) == expected_slots


# =================================================================================================
#  for_objectives
# =================================================================================================
def test_for_objectives_single_family(store: DistanceStore):
    """Two separation-family metrics yield one SeparationTracker, which is the main one."""
    # --- act --------------------------
    trackers = DiversityContributionTrackers.for_objectives(
        single_term_objective(DiversityMetric.GEOMEAN_SEPARATION),
        tie_breaker_objectives([DiversityMetric.NON_ZERO_SEPARATION_FRAC]),
        store,
    )

    # --- assert -----------------------
    assert len(trackers._trackers) == 1
    assert type(trackers.main) is SeparationTracker
    assert trackers.main is trackers._trackers[0]


def test_for_objectives_repeated_family(store: DistanceStore):
    """Tie-breakers repeating the main objective's family add no tracker."""
    # --- act --------------------------
    trackers = DiversityContributionTrackers.for_objectives(
        single_term_objective(DiversityMetric.MIN_SEPARATION),
        tie_breaker_objectives([DiversityMetric.MIN_SEPARATION, DiversityMetric.MEAN_SEPARATION]),
        store,
    )

    # --- assert -----------------------
    assert len(trackers._trackers) == 1
    assert type(trackers.main) is SeparationTracker


def test_more_pairs_than_the_unrolled_read_handles_are_rejected(store: DistanceStore):
    """`selected_contributions` is unrolled for MAX_CONTRIBUTION_SLOTS trackers, so a larger set is refused."""
    # --- arrange ----------------------
    distance_metrics = [DistanceMetric.l1_manhattan(), DistanceMetric.l2_euclidean(), DistanceMetric.linf_chebyshev()]
    trackers_by_pair = {
        (distance_metric, DiversityContributionFamily.SEPARATION): SeparationTracker(store)
        for distance_metric in distance_metrics[: MAX_CONTRIBUTION_SLOTS + 1]
    }

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at most"):
        DiversityContributionTrackers(trackers_by_pair, next(iter(trackers_by_pair)))


# =================================================================================================
#  Applying mutations, copy
# =================================================================================================
def test_mutations_reach_every_tracker(store: DistanceStore):
    # --- arrange ----------------------
    # hand-built two-family set, mirroring what a mixed-metric configuration would construct
    sep, mean = SeparationTracker(store), MeanDistanceTracker(store)
    trackers = DiversityContributionTrackers(
        trackers_by_pair={
            (None, DiversityContributionFamily.SEPARATION): sep,
            (None, DiversityContributionFamily.MEAN_DISTANCE): mean,
        },
        main_pair=(None, DiversityContributionFamily.MEAN_DISTANCE),
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
    trackers = DiversityContributionTrackers.for_objectives(
        single_term_objective(DiversityMetric.GEOMEAN_SEPARATION), [], store
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


# =================================================================================================
#  The tuple that selected_contributions returns
# =================================================================================================
def test_selected_contributions_one_array_per_slot(store: DistanceStore):
    """A single-family set fills its one slot with the selected vectors' separation values."""
    # --- arrange ----------------------
    trackers = DiversityContributionTrackers.for_objectives(
        single_term_objective(DiversityMetric.GEOMEAN_SEPARATION), [], store
    )
    trackers.add(np.int32(0))
    trackers.add(np.int32(2))  # selection: points 0.0 and 3.0 on a line
    selected = np.full(N, False, dtype=np.bool)
    selected[[0, 2]] = True
    selected_indices = np.array([0, 2], dtype=np.int32)

    # --- act --------------------------
    contributions = trackers.selected_contributions(selected, np.int32(2), selected_indices)

    # --- assert -----------------------
    assert len(contributions) == 1  # one array per tracked pair
    np.testing.assert_allclose(contributions[0], [3.0, 3.0])  # separation of the two selected points


def test_selected_contributions_slot_order_follows_the_pairs():
    """The arrays are returned in slot order, and each slot holds the separations computed with that pair's distance."""
    # --- arrange ----------------------
    # one family, two distances, on 2-D vectors where L1 and L2 disagree, so a swapped slot is detectable
    vectors_2d = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [10.0, 0.0]], dtype=np.float32)
    store_l1 = DistanceStore.full_matrix_from_vectors(vectors_2d, DistanceMetric.l1_manhattan())
    store_l2 = DistanceStore.full_matrix_from_vectors(vectors_2d, DistanceMetric.l2_euclidean())
    ref_l1, ref_l2 = SeparationTracker(store_l1), SeparationTracker(store_l2)
    trackers = DiversityContributionTrackers(
        trackers_by_pair={
            (DistanceMetric.l1_manhattan(), DiversityContributionFamily.SEPARATION): SeparationTracker(store_l1),
            (DistanceMetric.l2_euclidean(), DiversityContributionFamily.SEPARATION): SeparationTracker(store_l2),
        },
        main_pair=(DistanceMetric.l1_manhattan(), DiversityContributionFamily.SEPARATION),
    )
    selected = np.full(4, False, dtype=np.bool)
    selected[[0, 1, 2]] = True
    selected_indices = np.array([0, 1, 2], dtype=np.int32)
    for index in selected_indices:
        trackers.add(np.int32(index))
        ref_l1.add(np.int32(index))
        ref_l2.add(np.int32(index))

    # --- act --------------------------
    contributions = trackers.selected_contributions(selected, np.int32(3), selected_indices)

    # --- assert -----------------------
    assert len(contributions) == 2  # one slot per (distance, family) pair, in slot order
    np.testing.assert_allclose(
        contributions[0], ref_l1.contribution_wrt_selection(selected, np.int32(3))[selected_indices]
    )
    np.testing.assert_allclose(
        contributions[1], ref_l2.contribution_wrt_selection(selected, np.int32(3))[selected_indices]
    )
    assert not np.allclose(contributions[0], contributions[1])  # L1 and L2 disagree here, so the slots are distinct
