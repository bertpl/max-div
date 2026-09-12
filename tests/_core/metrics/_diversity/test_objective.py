import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjective,
    DiversityTerm,
    TermAggregation,
)


def _objective(*metrics: DiversityMetric) -> DiversityObjective:
    """Return an objective over one term per given metric."""
    return DiversityObjective(tuple(DiversityTerm(metric) for metric in metrics))


# =================================================================================================
#  Construction
# =================================================================================================
def test_empty_objective_is_rejected() -> None:
    """There is nothing to maximize without a term."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at least one term"):
        DiversityObjective(())


def test_an_objective_aggregates_hierarchically_unless_told_otherwise() -> None:
    """The primary objective's construction needs no kind; hierarchical is the default."""
    # --- act / assert -----------------
    assert _objective(DiversityMetric.MIN_SEPARATION).aggregation == TermAggregation.HIERARCHICAL


def test_a_flattened_objective_rejects_terms_with_different_metrics() -> None:
    """One metric reduces the concatenated rows, so every term must carry that metric."""
    # --- arrange ----------------------
    terms = (
        DiversityTerm(DiversityMetric.MIN_SEPARATION, DistanceMetric.l1_manhattan()),
        DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.l2_euclidean()),
    )

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="one metric"):
        DiversityObjective(terms, aggregation=TermAggregation.FLATTENED)
    DiversityObjective(terms)  # the same terms aggregate hierarchically without complaint


# =================================================================================================
#  Single-term accessors
# =================================================================================================
def test_main_metric_and_family_read_the_single_term() -> None:
    """The single term's metric and its contribution family are exposed."""
    # --- arrange ----------------------
    objective = _objective(DiversityMetric.MEAN_PAIRWISE_DISTANCE)

    # --- act / assert -----------------
    assert objective.main_diversity_metric == DiversityMetric.MEAN_PAIRWISE_DISTANCE
    assert objective.main_contribution_family == DiversityContributionFamily.MEAN_DISTANCE


@pytest.mark.parametrize(
    "metric, expected",
    [
        (DiversityMetric.GEOMEAN_SEPARATION, True),
        (DiversityMetric.MIN_SEPARATION, True),
        (DiversityMetric.MEAN_PAIRWISE_DISTANCE, False),
    ],
)
def test_single_separation_term_predicate(metric: DiversityMetric, expected: bool) -> None:
    """One separation-family term is a single separation objective; a mean-distance term is not."""
    # --- act / assert -----------------
    assert _objective(metric).has_single_separation_term is expected


def test_a_two_term_objective_is_not_a_single_separation_term() -> None:
    """More than one term is never a single separation objective, whatever the families."""
    # --- act / assert -----------------
    objective = _objective(DiversityMetric.MIN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION)
    assert not objective.has_single_separation_term


# =================================================================================================
#  Derived facts
# =================================================================================================
def test_contribution_keys_are_distinct_and_first_seen_ordered() -> None:
    """Terms sharing a family and a distance share a key; a different distance is a key of its own."""
    # --- arrange ----------------------
    objective = DiversityObjective(
        (
            DiversityTerm(DiversityMetric.MEAN_PAIRWISE_DISTANCE),
            DiversityTerm(DiversityMetric.MIN_SEPARATION, DistanceMetric.l1_manhattan()),
            DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.l1_manhattan()),
            DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.l2_euclidean()),
        )
    )

    # --- act / assert -----------------
    assert objective.contribution_keys == (
        (DiversityContributionFamily.MEAN_DISTANCE, None),
        (DiversityContributionFamily.SEPARATION, DistanceMetric.l1_manhattan()),
        (DiversityContributionFamily.SEPARATION, DistanceMetric.l2_euclidean()),
    )


# =================================================================================================
#  Tie-breakers
# =================================================================================================
def test_a_tie_breaker_is_flattened_over_the_objectives_distances() -> None:
    """The tie-breaker pairs its metric with each distinct distance of the terms, in first-seen order."""
    # --- arrange ----------------------
    objective = DiversityObjective(
        (
            DiversityTerm(DiversityMetric.MIN_SEPARATION, DistanceMetric.l2_euclidean()),
            DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.along_axis(0)),
            DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.l2_euclidean()),
        )
    )

    # --- act --------------------------
    tie_breaker = objective.tie_breaker(DiversityMetric.NON_ZERO_SEPARATION_FRAC)

    # --- assert -----------------------
    assert tie_breaker == DiversityObjective(
        (
            DiversityTerm(DiversityMetric.NON_ZERO_SEPARATION_FRAC, DistanceMetric.l2_euclidean()),
            DiversityTerm(DiversityMetric.NON_ZERO_SEPARATION_FRAC, DistanceMetric.along_axis(0)),
        ),
        aggregation=TermAggregation.FLATTENED,
    )


def test_a_tie_breaker_on_a_one_term_objective_shares_its_distance() -> None:
    """On a native problem the tie-breaker is one term over the same (given) distance as the objective."""
    # --- act --------------------------
    tie_breaker = _objective(DiversityMetric.MIN_SEPARATION).tie_breaker(DiversityMetric.MEAN_SEPARATION)

    # --- assert -----------------------
    assert tie_breaker.terms == (DiversityTerm(DiversityMetric.MEAN_SEPARATION, None),)
    assert tie_breaker.aggregation == TermAggregation.FLATTENED


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
def test_default_tie_breakers_follow_the_main_metric(metric: DiversityMetric, expected: list[DiversityMetric]) -> None:
    """Near-degenerate main metrics get separating tie-breakers, each a flattened objective; the rest get none."""
    # --- arrange ----------------------
    objective = _objective(metric)

    # --- act / assert -----------------
    assert objective.default_tie_breakers == [objective.tie_breaker(tie_breaker) for tie_breaker in expected]
