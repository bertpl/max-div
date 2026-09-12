import numpy as np
import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjectiveHybridFlattened,
    DiversityObjectiveHybridGeoMean,
    DiversityObjectiveSimple,
    DiversityTrackerSpec,
)

SEPARATION = DiversityContributionFamily.SEPARATION
MEAN_DISTANCE = DiversityContributionFamily.MEAN_DISTANCE
L1 = DistanceMetric.l1_manhattan()
L2 = DistanceMetric.l2_euclidean()


def _f32(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# =================================================================================================
#  Construction
# =================================================================================================
def test_a_simple_objective_defaults_its_distance_to_the_problems_own() -> None:
    """A simple objective built without a distance reads the problem's own distance (`None`)."""
    # --- act / assert -----------------
    assert DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION).distance_metric is None


def test_a_geomean_hybrid_needs_at_least_two_terms() -> None:
    """One term is a `DiversityObjectiveSimple`, so a geometric-mean hybrid rejects fewer than two."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at least two terms"):
        DiversityObjectiveHybridGeoMean((DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION),))


# =================================================================================================
#  tracker_specs (and the facts derived from it)
# =================================================================================================
@pytest.mark.parametrize(
    "objective, expected_specs",
    [
        pytest.param(
            DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE),
            (DiversityTrackerSpec(None, MEAN_DISTANCE),),
            id="simple",
        ),
        pytest.param(
            DiversityObjectiveHybridFlattened(DiversityMetric.MIN_SEPARATION, (L1, L2)),
            (DiversityTrackerSpec(L1, SEPARATION), DiversityTrackerSpec(L2, SEPARATION)),
            id="flattened_one_family_two_distances",
        ),
        pytest.param(
            DiversityObjectiveHybridGeoMean(
                (
                    DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE),  # (None, MEAN_DISTANCE)
                    DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),  # (L1, SEPARATION)
                    DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L1),  # repeat: (L1, SEPARATION)
                )
            ),
            (DiversityTrackerSpec(None, MEAN_DISTANCE), DiversityTrackerSpec(L1, SEPARATION)),
            id="geomean_dedups_in_first_seen_order",
        ),
    ],
)
def test_tracker_specs(objective, expected_specs) -> None:
    """The distinct specs an objective reads, in first-seen order, each once."""
    # --- act / assert -----------------
    assert objective.tracker_specs() == expected_specs


@pytest.mark.parametrize(
    "objective, expected",
    [
        (DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION), (None,)),
        (DiversityObjectiveHybridFlattened(DiversityMetric.MIN_SEPARATION, (L1, L2)), (L1, L2)),
        (
            DiversityObjectiveHybridGeoMean(
                (
                    DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                    DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
                )
            ),
            (L1, L2),
        ),
    ],
)
def test_distinct_distance_metrics(objective, expected) -> None:
    """The distinct distance metrics an objective reads, in first-seen order."""
    # --- act / assert -----------------
    assert objective.distinct_distance_metrics() == expected


@pytest.mark.parametrize(
    "objective, expected",
    [
        (DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION), True),
        (DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE), False),  # mean-distance family
        (DiversityObjectiveHybridFlattened(DiversityMetric.MIN_SEPARATION, (L1, L2)), False),  # two specs
    ],
)
def test_reads_single_separation_tracker(objective, expected) -> None:
    """One separation spec is the batched-init case; a second spec or another family is not."""
    # --- act / assert -----------------
    assert objective.reads_single_separation_tracker() is expected


# =================================================================================================
#  compute
# =================================================================================================
def test_simple_computes_its_metric_over_its_one_spec() -> None:
    """A simple objective reduces its one spec's contribution array with its diversity metric."""
    # --- arrange ----------------------
    objective = DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION)
    contributions = {DiversityTrackerSpec(None, SEPARATION): _f32([4.0, 2.0, 6.0])}

    # --- act / assert -----------------
    assert objective.compute(contributions) == pytest.approx(2.0)  # min of the separation array


def test_flattened_computes_its_metric_over_its_joined_specs() -> None:
    """A flattened objective reduces the concatenation of all its specs' arrays with its diversity metric."""
    # --- arrange ----------------------
    objective = DiversityObjectiveHybridFlattened(DiversityMetric.MIN_SEPARATION, (L1, L2))
    contributions = {
        DiversityTrackerSpec(L1, SEPARATION): _f32([5.0, 3.0]),
        DiversityTrackerSpec(L2, SEPARATION): _f32([2.0, 4.0]),
    }

    # --- act / assert -----------------
    assert objective.compute(contributions) == pytest.approx(2.0)  # min over [5, 3, 2, 4]


def test_geomean_computes_the_geometric_mean_of_its_terms() -> None:
    """A geometric-mean hybrid returns the geometric mean of its terms' diversity scores."""
    # --- arrange ----------------------
    objective = DiversityObjectiveHybridGeoMean(
        (
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
        )
    )
    contributions = {
        DiversityTrackerSpec(L1, SEPARATION): _f32([4.0, 8.0]),  # term 1 min = 4
        DiversityTrackerSpec(L2, SEPARATION): _f32([9.0, 3.0]),  # term 2 min = 3
    }

    # --- act / assert -----------------
    assert objective.compute(contributions) == pytest.approx(np.sqrt(12.0))  # geometric mean of 4 and 3


# =================================================================================================
#  default_tie_breakers
# =================================================================================================
@pytest.mark.parametrize(
    "diversity_metric, expected_tie_breaker_metrics",
    [
        (
            DiversityMetric.MIN_SEPARATION,
            [DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC],
        ),
        (DiversityMetric.GEOMEAN_SEPARATION, [DiversityMetric.NON_ZERO_SEPARATION_FRAC]),
        (DiversityMetric.APPROX_GEOMEAN_SEPARATION, [DiversityMetric.NON_ZERO_SEPARATION_FRAC]),
        (DiversityMetric.MEAN_SEPARATION, []),
        (DiversityMetric.MEAN_PAIRWISE_DISTANCE, []),
    ],
)
def test_a_simple_objectives_default_tie_breakers_follow_its_metric(
    diversity_metric, expected_tie_breaker_metrics
) -> None:
    """A near-degenerate diversity metric gets separating tie-breakers over its own distance; the rest get none."""
    # --- act --------------------------
    tie_breakers = DiversityObjectiveSimple(diversity_metric, L2).default_tie_breakers()

    # --- assert -----------------------
    assert tie_breakers == [DiversityObjectiveHybridFlattened(metric, (L2,)) for metric in expected_tie_breaker_metrics]


def test_a_geomean_hybrids_default_tie_breakers_span_its_distances() -> None:
    """A geometric-mean hybrid gets the separating pair, each flattened over its distinct distance metrics."""
    # --- arrange ----------------------
    objective = DiversityObjectiveHybridGeoMean(
        (
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
            DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L2),
        )
    )

    # --- act / assert -----------------
    assert objective.default_tie_breakers() == [
        DiversityObjectiveHybridFlattened(DiversityMetric.APPROX_GEOMEAN_SEPARATION, (L1, L2)),
        DiversityObjectiveHybridFlattened(DiversityMetric.NON_ZERO_SEPARATION_FRAC, (L1, L2)),
    ]


def test_a_flattened_objective_has_no_tie_breakers() -> None:
    """A tie-breaker is not itself ranked by further tie-breakers."""
    # --- act / assert -----------------
    assert DiversityObjectiveHybridFlattened(DiversityMetric.MEAN_SEPARATION, (None,)).default_tie_breakers() == []
