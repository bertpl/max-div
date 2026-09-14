import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjectiveHybrid,
    DiversityObjectiveSimple,
    DiversityTrackerSpec,
    HybridCombinationType,
)
from max_div._core.solver._diversity_contribution import DiversityObjectiveBindings

SEPARATION = DiversityContributionFamily.SEPARATION
MEAN_DISTANCE = DiversityContributionFamily.MEAN_DISTANCE
L1 = DistanceMetric.l1_manhattan()
L2 = DistanceMetric.l2_euclidean()


@pytest.mark.parametrize(
    "diversity_objectives, expected_distance_metrics, expected_tracker_specs, expected_positions",
    [
        pytest.param(
            [DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION)],
            (None,),
            (DiversityTrackerSpec(None, SEPARATION),),
            ((0,),),
            id="one_simple_objective_over_the_problems_own_distance",
        ),
        pytest.param(
            [
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION),
                DiversityObjectiveSimple(DiversityMetric.APPROX_GEOMEAN_SEPARATION),
                DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE),
            ],
            (None,),
            (DiversityTrackerSpec(None, SEPARATION), DiversityTrackerSpec(None, MEAN_DISTANCE)),
            ((0,), (0,), (1,)),
            id="tie_breakers_sharing_the_primary_spec_add_no_tracker",
        ),
        pytest.param(
            [
                DiversityObjectiveHybrid(
                    (
                        DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
                        DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                        DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L2),  # repeats the L2 spec
                    )
                ),
                DiversityObjectiveHybrid(
                    (
                        DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L1),
                        DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L2),
                    ),
                    HybridCombinationType.ARITHMETIC_MEAN,
                ),
                DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE, L1),
            ],
            (L2, L1),
            (
                DiversityTrackerSpec(L2, SEPARATION),
                DiversityTrackerSpec(L1, SEPARATION),
                DiversityTrackerSpec(L1, MEAN_DISTANCE),
            ),
            ((0, 1, 0), (1, 0), (2,)),  # the hybrid repeats the L2 spec
            id="hybrid_over_two_distances_with_a_hybrid_tie_breaker",
        ),
    ],
)
def test_for_objectives(diversity_objectives, expected_distance_metrics, expected_tracker_specs, expected_positions):
    """Distance metrics and specs are distinct in first-seen order; each objective's positions follow its spec order."""
    # --- act --------------------------
    bindings = DiversityObjectiveBindings.for_objectives(diversity_objectives)

    # --- assert -----------------------
    assert bindings.distance_metrics == expected_distance_metrics
    assert bindings.tracker_specs == expected_tracker_specs
    assert bindings.objective_spec_positions == expected_positions
