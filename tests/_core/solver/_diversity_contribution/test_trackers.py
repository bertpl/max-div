import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric, DiversityContributionFamily, DiversityMetric
from max_div._core.metrics._distance import compute_pdist
from max_div._core.solver._diversity_contribution import (
    DiversityContributionTrackers,
    MeanDistanceTracker,
    SeparationTracker,
)

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 6


@pytest.fixture
def pdist() -> np.ndarray:
    vectors = np.array([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0]], dtype=np.float32)
    return compute_pdist(vectors, DistanceMetric.L1_MANHATTAN)


# =================================================================================================
#  Tests
# =================================================================================================
def test_for_metrics_single_family(pdist: np.ndarray):
    # --- act ---------------------------------------------
    trackers = DiversityContributionTrackers.for_metrics(
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[DiversityMetric.NON_ZERO_SEPARATION_FRAC],
        pdist=pdist,
        n=np.int32(N),
    )

    # --- assert ------------------------------------------
    # both metrics are separation-family -> exactly one tracker, which is also the main one
    assert len(trackers._trackers) == 1
    assert type(trackers.main) is SeparationTracker
    assert trackers.main is trackers._trackers[0]


def test_for_metrics_mixed_families(pdist: np.ndarray):
    # --- act ---------------------------------------------
    trackers = DiversityContributionTrackers.for_metrics(
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[DiversityMetric.MIN_SEPARATION, DiversityMetric.MEAN_SEPARATION],
        pdist=pdist,
        n=np.int32(N),
    )

    # --- assert ------------------------------------------
    # duplicate families are deduplicated
    assert len(trackers._trackers) == 1
    assert type(trackers.main) is SeparationTracker


def test_mutations_fan_out_to_all_trackers(pdist: np.ndarray):
    # --- arrange -----------------------------------------
    # hand-built two-family set, mirroring what a mixed-metric configuration would construct
    sep, mean = SeparationTracker(pdist, np.int32(N)), MeanDistanceTracker(pdist, np.int32(N))
    trackers = DiversityContributionTrackers(
        trackers_by_family={
            DiversityContributionFamily.SEPARATION: sep,
            DiversityContributionFamily.MEAN_DISTANCE: mean,
        },
        main_family=DiversityContributionFamily.MEAN_DISTANCE,
    )
    sep_ref, mean_ref = SeparationTracker(pdist, np.int32(N)), MeanDistanceTracker(pdist, np.int32(N))
    selected = np.full(N, False, dtype=np.bool)
    selected[[0, 3]] = True

    # --- act ---------------------------------------------
    trackers.add(np.int32(0))
    trackers.add_many(np.array([2, 3], dtype=np.int32))
    trackers.set_snapshot()
    trackers.remove_many(np.array([2, 3], dtype=np.int32), new_selection=np.array([0], dtype=np.int32))
    trackers.restore_snapshot()
    trackers.remove(np.int32(2), new_selection=np.array([0, 3], dtype=np.int32))

    for ref in (sep_ref, mean_ref):
        ref.add(np.int32(0))
        ref.add_many(np.array([2, 3], dtype=np.int32))
        ref.remove(np.int32(2), new_selection=np.array([0, 3], dtype=np.int32))

    # --- assert ------------------------------------------
    assert trackers.main is mean
    np.testing.assert_array_equal(
        sep.contribution_wrt_selection(selected, np.int32(2)),
        sep_ref.contribution_wrt_selection(selected, np.int32(2)),
    )
    np.testing.assert_array_equal(
        mean.contribution_wrt_selection(selected, np.int32(2)),
        mean_ref.contribution_wrt_selection(selected, np.int32(2)),
    )


def test_copy_is_independent(pdist: np.ndarray):
    # --- arrange -----------------------------------------
    trackers = DiversityContributionTrackers.for_metrics(
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[],
        pdist=pdist,
        n=np.int32(N),
    )
    trackers.add(np.int32(0))
    clone = trackers.copy()
    selected = np.full(N, False, dtype=np.bool)
    selected[0] = True
    before = clone.main.contribution_wrt_selection(selected, np.int32(1)).copy()

    # --- act ---------------------------------------------
    trackers.add(np.int32(4))

    # --- assert ------------------------------------------
    assert clone.main is not trackers.main
    np.testing.assert_array_equal(clone.main.contribution_wrt_selection(selected, np.int32(1)), before)
