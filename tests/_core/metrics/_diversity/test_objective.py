import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjective,
    DiversityTerm,
    TermAggregationType,
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


def test_aggregation_type_defaults_to_the_geometric_mean_of_terms() -> None:
    """`aggregation_type` defaults to `GEOMEAN_OF_TERMS`."""
    # --- act / assert -----------------
    assert _objective(DiversityMetric.MIN_SEPARATION).aggregation_type == TermAggregationType.GEOMEAN_OF_TERMS


def test_a_flattened_objective_rejects_terms_with_different_metrics() -> None:
    """A FLATTENED_TERMS objective is computed with one metric, so every term must use it."""
    # --- arrange ----------------------
    terms = (
        DiversityTerm(DiversityMetric.MIN_SEPARATION, DistanceMetric.l1_manhattan()),
        DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.l2_euclidean()),
    )

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="one diversity metric"):
        DiversityObjective(terms, aggregation_type=TermAggregationType.FLATTENED_TERMS)


# =================================================================================================
#  Single-term accessors
# =================================================================================================
def test_main_diversity_metric_and_family_read_the_single_term() -> None:
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
    """Terms sharing a family and a distance share a key; a term with a different distance gets a key of its own."""
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
@pytest.mark.parametrize(
    "terms, expected_distance_metrics",
    [
        # one term with no distance of its own (None, the problem's distance): the tie-breaker is one term over it
        ((DiversityTerm(DiversityMetric.MIN_SEPARATION),), [None]),
        # several terms: one tie-breaker term per distinct distance, in first-seen order
        (
            (
                DiversityTerm(DiversityMetric.MIN_SEPARATION, DistanceMetric.l2_euclidean()),
                DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.along_axis(0)),
                DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.l2_euclidean()),
            ),
            [DistanceMetric.l2_euclidean(), DistanceMetric.along_axis(0)],
        ),
    ],
    ids=["one_term", "several_terms"],
)
def test_a_tie_breaker_is_flattened_over_the_distinct_distances(
    terms: tuple[DiversityTerm, ...], expected_distance_metrics: list[DistanceMetric | None]
) -> None:
    """The tie-breaker has one term per distinct distance of the source terms, each using the tie-breaker metric."""
    # --- act --------------------------
    tie_breaker = DiversityObjective(terms).build_tie_breaker(DiversityMetric.NON_ZERO_SEPARATION_FRAC)

    # --- assert -----------------------
    assert tie_breaker == DiversityObjective(
        tuple(DiversityTerm(DiversityMetric.NON_ZERO_SEPARATION_FRAC, d) for d in expected_distance_metrics),
        aggregation_type=TermAggregationType.FLATTENED_TERMS,
    )


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
def test_default_tie_breakers_follow_the_main_diversity_metric(
    metric: DiversityMetric, expected: list[DiversityMetric]
) -> None:
    """Near-degenerate main metrics get separating tie-breakers built by `build_tie_breaker`; the rest get none."""
    # --- arrange ----------------------
    objective = _objective(metric)

    # --- act / assert -----------------
    assert objective.default_tie_breakers == [objective.build_tie_breaker(tb) for tb in expected]
