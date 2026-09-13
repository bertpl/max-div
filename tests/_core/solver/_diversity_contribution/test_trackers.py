import numpy as np
import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjectiveHybridFlattened,
    DiversityTrackerSpec,
)
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._diversity_contribution import (
    DiversityContributionTrackers,
    MeanDistanceTracker,
    SeparationTracker,
)
from tests._core.solver.objectives import simple_objective, tie_breaker_objectives

SEPARATION = DiversityContributionFamily.SEPARATION
MEAN_DISTANCE = DiversityContributionFamily.MEAN_DISTANCE

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 6
_VECTORS = np.array([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0]], dtype=np.float32)


@pytest.fixture
def store() -> DistanceStore:
    return DistanceStore.full_matrix_from_vectors(_VECTORS, DistanceMetric.l1_manhattan())


# =================================================================================================
#  for_objectives
# =================================================================================================
def test_for_objectives_single_family(store: DistanceStore):
    """Two separation-family metrics yield one SeparationTracker, which represents the main objective."""
    # --- arrange ----------------------
    main_objective = simple_objective(DiversityMetric.GEOMEAN_SEPARATION)

    # --- act --------------------------
    trackers = DiversityContributionTrackers.for_objectives(
        [main_objective, *tie_breaker_objectives([DiversityMetric.NON_ZERO_SEPARATION_FRAC])], store
    )

    # --- assert -----------------------
    assert len(trackers._trackers) == 1
    assert type(trackers.tracker_for(main_objective)) is SeparationTracker
    assert trackers.tracker_for(main_objective) is trackers._trackers[0]


def test_for_objectives_repeated_family(store: DistanceStore):
    """Tie-breakers repeating the main objective's family add no tracker."""
    # --- act --------------------------
    trackers = DiversityContributionTrackers.for_objectives(
        [
            simple_objective(DiversityMetric.MIN_SEPARATION),
            *tie_breaker_objectives([DiversityMetric.MIN_SEPARATION, DiversityMetric.MEAN_SEPARATION]),
        ],
        store,
    )

    # --- assert -----------------------
    assert len(trackers._trackers) == 1
    assert type(trackers._trackers[0]) is SeparationTracker


# =================================================================================================
#  Applying mutations
# =================================================================================================
def test_mutations_reach_every_tracker(store: DistanceStore):
    # --- arrange ----------------------
    # hand-built two-family set, mirroring what a mixed-metric configuration would construct
    sep, mean = SeparationTracker(store), MeanDistanceTracker(store)
    trackers = DiversityContributionTrackers(
        trackers_by_spec={
            DiversityTrackerSpec(None, SEPARATION): sep,
            DiversityTrackerSpec(None, MEAN_DISTANCE): mean,
        }
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
    np.testing.assert_array_equal(
        sep.contribution_wrt_selection(selected, np.int32(2)),
        sep_ref.contribution_wrt_selection(selected, np.int32(2)),
    )
    np.testing.assert_array_equal(
        mean.contribution_wrt_selection(selected, np.int32(2)),
        mean_ref.contribution_wrt_selection(selected, np.int32(2)),
    )


# =================================================================================================
#  selected_contributions
# =================================================================================================
def test_selected_contributions_one_array_per_spec(store: DistanceStore):
    """A single-family set returns one spec's array, the selected vectors' separation values."""
    # --- arrange ----------------------
    trackers = DiversityContributionTrackers.for_objectives(
        [simple_objective(DiversityMetric.GEOMEAN_SEPARATION)], store
    )
    trackers.add(np.int32(0))
    trackers.add(np.int32(2))  # selection: points 0.0 and 3.0 on a line
    selected = np.full(N, False, dtype=np.bool)
    selected[[0, 2]] = True
    selected_indices = np.array([0, 2], dtype=np.int32)

    # --- act --------------------------
    contributions = trackers.selected_contributions(selected, np.int32(2), selected_indices)

    # --- assert -----------------------
    assert trackers.tracker_specs == (DiversityTrackerSpec(None, SEPARATION),)  # one tracked spec
    assert len(contributions) == 1
    np.testing.assert_allclose(contributions[0], [3.0, 3.0])


def test_selected_contributions_orders_the_arrays_as_the_specs():
    """Each spec's array holds the separations computed with that spec's distance metric, at that spec's position."""
    # --- arrange ----------------------
    # one family, two distances, on 2-D vectors where L1 and L2 disagree, so swapped positions are detectable
    vectors_2d = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [10.0, 0.0]], dtype=np.float32)
    store_l1 = DistanceStore.full_matrix_from_vectors(vectors_2d, DistanceMetric.l1_manhattan())
    store_l2 = DistanceStore.full_matrix_from_vectors(vectors_2d, DistanceMetric.l2_euclidean())
    spec_l1 = DiversityTrackerSpec(DistanceMetric.l1_manhattan(), SEPARATION)
    spec_l2 = DiversityTrackerSpec(DistanceMetric.l2_euclidean(), SEPARATION)
    ref_l1, ref_l2 = SeparationTracker(store_l1), SeparationTracker(store_l2)
    trackers = DiversityContributionTrackers(
        trackers_by_spec={spec_l1: SeparationTracker(store_l1), spec_l2: SeparationTracker(store_l2)}
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
    assert trackers.tracker_specs == (spec_l1, spec_l2)
    np.testing.assert_allclose(
        contributions[0], ref_l1.contribution_wrt_selection(selected, np.int32(3))[selected_indices]
    )
    np.testing.assert_allclose(
        contributions[1], ref_l2.contribution_wrt_selection(selected, np.int32(3))[selected_indices]
    )
    assert not np.allclose(contributions[0], contributions[1])  # L1 and L2 disagree here


# =================================================================================================
#  tracker_for
# =================================================================================================
def test_tracker_for_refuses_an_objective_over_several_trackers(store: DistanceStore):
    """No single tracker represents an objective that reads several specs."""
    # --- arrange ----------------------
    flattened = DiversityObjectiveHybridFlattened(
        DiversityMetric.MIN_SEPARATION, (DistanceMetric.l1_manhattan(), DistanceMetric.l2_euclidean())
    )
    trackers = DiversityContributionTrackers.for_objectives([flattened], store)

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="No single tracker"):
        trackers.tracker_for(flattened)
