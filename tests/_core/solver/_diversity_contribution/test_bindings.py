import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjectiveSimple,
    DiversityTrackerSpec,
    HybridAggregationArithmeticMean,
)
from max_div._core.metrics._distance import FullMatrixDistanceSpec, VectorDistanceSpec
from max_div._core.solver._diversity_contribution import DiversityObjectiveBindings
from tests.helpers import hybrid_objective

SEPARATION = DiversityContributionFamily.SEPARATION
MEAN_DISTANCE = DiversityContributionFamily.MEAN_DISTANCE
USER_DISTANCES = FullMatrixDistanceSpec(matrix_id=0, label="user distances")
L1 = VectorDistanceSpec(matrix_id=0, metric=DistanceMetric.l1_manhattan(), is_matrix_preprocessed=True)
L2 = VectorDistanceSpec(matrix_id=0, metric=DistanceMetric.l2_euclidean(), is_matrix_preprocessed=True)


@pytest.mark.parametrize(
    "diversity_objectives, expected_distance_specs, expected_tracker_specs, expected_positions",
    [
        pytest.param(
            [DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, USER_DISTANCES)],
            (USER_DISTANCES,),
            (DiversityTrackerSpec(USER_DISTANCES, SEPARATION),),
            ((0,),),
            id="one_simple_objective_over_the_users_distances",
        ),
        pytest.param(
            [
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, USER_DISTANCES),
                DiversityObjectiveSimple(DiversityMetric.APPROX_GEOMEAN_SEPARATION, USER_DISTANCES),
                DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE, USER_DISTANCES),
            ],
            (USER_DISTANCES,),
            (DiversityTrackerSpec(USER_DISTANCES, SEPARATION), DiversityTrackerSpec(USER_DISTANCES, MEAN_DISTANCE)),
            ((0,), (0,), (1,)),
            id="tie_breakers_sharing_the_primary_spec_add_no_tracker",
        ),
        pytest.param(
            [
                hybrid_objective(
                    DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
                    DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                    DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L2),  # this term repeats the L2 spec
                ),
                hybrid_objective(
                    DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L1),
                    DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L2),
                    aggregation_type=HybridAggregationArithmeticMean,
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
def test_for_objectives(diversity_objectives, expected_distance_specs, expected_tracker_specs, expected_positions):
    """Distance and tracker specs are distinct in first-seen order; each objective's positions follow its spec order."""
    # --- act --------------------------
    bindings = DiversityObjectiveBindings.for_objectives(diversity_objectives)

    # --- assert -----------------------
    assert bindings.distance_specs == expected_distance_specs
    assert bindings.tracker_specs == expected_tracker_specs
    assert bindings.objective_spec_positions == expected_positions
