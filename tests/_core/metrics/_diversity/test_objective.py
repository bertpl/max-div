import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjectiveHybridFlattened,
    DiversityObjectiveHybridGeoMean,
    DiversityObjectiveSimple,
    default_tie_breaker_metrics,
    scoring_metric,
)

SEPARATION = DiversityContributionFamily.SEPARATION
MEAN_DISTANCE = DiversityContributionFamily.MEAN_DISTANCE
L1 = DistanceMetric.l1_manhattan()
L2 = DistanceMetric.l2_euclidean()


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
#  distance_family_pairs (and the facts derived from it)
# =================================================================================================
@pytest.mark.parametrize(
    "objective, expected_pairs",
    [
        pytest.param(
            DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE),
            ((None, MEAN_DISTANCE),),
            id="simple",
        ),
        pytest.param(
            DiversityObjectiveHybridFlattened(DiversityMetric.MIN_SEPARATION, (L1, L2)),
            ((L1, SEPARATION), (L2, SEPARATION)),
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
            ((None, MEAN_DISTANCE), (L1, SEPARATION)),
            id="geomean_dedups_in_first_seen_order",
        ),
    ],
)
def test_distance_family_pairs(objective, expected_pairs) -> None:
    """The distinct (distance, family) pairs an objective reads, in first-seen order, each once."""
    # --- act / assert -----------------
    assert objective.distance_family_pairs() == expected_pairs


@pytest.mark.parametrize(
    "objective, expected",
    [
        (DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION), True),
        (DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE), False),  # mean-distance family
        (DiversityObjectiveHybridFlattened(DiversityMetric.MIN_SEPARATION, (L1, L2)), False),  # two trackers
    ],
)
def test_has_single_separation_tracker(objective, expected) -> None:
    """One separation tracker is the batched-init case; a second tracker or another family is not."""
    # --- act / assert -----------------
    assert objective.has_single_separation_tracker() is expected


# =================================================================================================
#  Tie-breakers
# =================================================================================================
def test_build_tie_breaker_is_flattened_over_the_objectives_distances() -> None:
    """The tie-breaker carries its metric over the distinct distances the source objective reads."""
    # --- arrange ----------------------
    objective = DiversityObjectiveHybridGeoMean(
        (
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
            DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.along_axis(0)),
            DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L2),  # repeat of L2
        )
    )

    # --- act --------------------------
    tie_breaker = objective.build_tie_breaker(DiversityMetric.NON_ZERO_SEPARATION_FRAC)

    # --- assert -----------------------
    assert tie_breaker == DiversityObjectiveHybridFlattened(
        DiversityMetric.NON_ZERO_SEPARATION_FRAC, (L2, DistanceMetric.along_axis(0))
    )


def test_build_tie_breaker_on_a_simple_objective_shares_its_distance() -> None:
    """A tie-breaker of a simple objective is flattened over that objective's one distance."""
    # --- act --------------------------
    tie_breaker = DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION).build_tie_breaker(
        DiversityMetric.MEAN_SEPARATION
    )

    # --- assert -----------------------
    assert tie_breaker == DiversityObjectiveHybridFlattened(DiversityMetric.MEAN_SEPARATION, (None,))


@pytest.mark.parametrize(
    "metric, expected",
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
def test_default_tie_breaker_metrics_follow_the_main_metric(metric, expected) -> None:
    """Near-degenerate main metrics get separating tie-breakers; the rest get none."""
    # --- act / assert -----------------
    assert default_tie_breaker_metrics(metric) == expected


# =================================================================================================
#  scoring_metric
# =================================================================================================
def test_scoring_metric_reads_a_single_metric_objective() -> None:
    """A simple or flattened objective is scored by its one diversity metric."""
    # --- act / assert -----------------
    assert scoring_metric(DiversityObjectiveSimple(DiversityMetric.MEAN_SEPARATION)) == DiversityMetric.MEAN_SEPARATION
    flattened = DiversityObjectiveHybridFlattened(DiversityMetric.MIN_SEPARATION, (L1, L2))
    assert scoring_metric(flattened) == DiversityMetric.MIN_SEPARATION


def test_scoring_metric_rejects_a_geomean_hybrid() -> None:
    """A geometric-mean hybrid combines several metrics, so it has no single scoring metric."""
    # --- arrange ----------------------
    objective = DiversityObjectiveHybridGeoMean(
        (
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION),
            DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION),
        )
    )

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="no single scoring metric"):
        scoring_metric(objective)
