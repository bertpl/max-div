import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjective,
    DiversityTerm,
)


def _objective(*metrics: DiversityMetric) -> DiversityObjective:
    """Return an objective over one term per given metric, each over the given distance."""
    return DiversityObjective(tuple(DiversityTerm(metric) for metric in metrics))


def test_empty_objective_is_rejected() -> None:
    """There is nothing to maximize without a term."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at least one term"):
        DiversityObjective(())


def test_main_metric_and_family_read_the_single_term() -> None:
    """The single term's metric and its contribution family are exposed."""
    # --- arrange ----------------------
    objective = _objective(DiversityMetric.MEAN_PAIRWISE_DISTANCE)

    # --- act / assert -----------------
    assert objective.main_metric == DiversityMetric.MEAN_PAIRWISE_DISTANCE
    assert objective.main_family == DiversityContributionFamily.MEAN_DISTANCE


def test_contribution_families_are_distinct_and_first_seen_ordered() -> None:
    """Repeated families collapse to one entry, in the order the terms first use them."""
    # --- arrange ----------------------
    objective = DiversityObjective(
        (
            DiversityTerm(DiversityMetric.MEAN_PAIRWISE_DISTANCE),
            DiversityTerm(DiversityMetric.MIN_SEPARATION, DistanceMetric.l1_manhattan()),
            DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION, DistanceMetric.l2_euclidean()),
        )
    )

    # --- act / assert -----------------
    assert objective.contribution_families == (
        DiversityContributionFamily.MEAN_DISTANCE,
        DiversityContributionFamily.SEPARATION,
    )


@pytest.mark.parametrize(
    "metric, expected",
    [
        (DiversityMetric.GEOMEAN_SEPARATION, True),
        (DiversityMetric.MIN_SEPARATION, True),
        (DiversityMetric.MEAN_PAIRWISE_DISTANCE, False),
    ],
)
def test_single_separation_tracker_predicate(metric: DiversityMetric, expected: bool) -> None:
    """One separation-family term is served by a single separation tracker; a mean-distance term is not."""
    # --- act / assert -----------------
    assert _objective(metric).uses_single_separation_tracker is expected


def test_a_two_term_objective_is_not_a_single_separation_tracker() -> None:
    """More than one term needs more than one tracker, whatever the families."""
    # --- act / assert -----------------
    objective = _objective(DiversityMetric.MIN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION)
    assert not objective.uses_single_separation_tracker


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
    """Near-degenerate main metrics get separating tie-breakers; the rest get none."""
    # --- act / assert -----------------
    assert _objective(metric).default_tie_breakers == expected
