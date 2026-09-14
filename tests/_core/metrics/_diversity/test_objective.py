import numpy as np
import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjectiveHybrid,
    DiversityObjectiveSimple,
    DiversityTrackerSpec,
    HybridCombination,
)

SEPARATION = DiversityContributionFamily.SEPARATION
MEAN_DISTANCE = DiversityContributionFamily.MEAN_DISTANCE
L1 = DistanceMetric.l1_manhattan()
L2 = DistanceMetric.l2_euclidean()


def _f32(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _hybrid(*terms: DiversityObjectiveSimple, combination=HybridCombination.GEOMETRIC_MEAN) -> DiversityObjectiveHybrid:
    """Build a `DiversityObjectiveHybrid` from loose terms, geometric-mean by default."""
    return DiversityObjectiveHybrid(terms, combination)


# =================================================================================================
#  Construction
# =================================================================================================
def test_a_simple_objective_defaults_its_distance_to_the_problems_own() -> None:
    """A simple objective built without a distance reads the problem's own distance (`None`)."""
    # --- act / assert -----------------
    assert DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION).distance_metric is None


def test_a_hybrid_needs_at_least_two_terms() -> None:
    """One term is a `DiversityObjectiveSimple`, so a hybrid rejects fewer than two."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at least two terms"):
        _hybrid(DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION))


def test_a_hybrid_rejects_a_term_that_is_not_a_simple_objective() -> None:
    """A term must be a simple objective, so that each term reads exactly one array."""
    # --- arrange ----------------------
    simple = DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1)
    hybrid = _hybrid(simple, DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2))

    # --- act / assert -----------------
    with pytest.raises(TypeError, match="simple objectives"):
        _hybrid(simple, hybrid)  # ty: ignore[invalid-argument-type]


def test_a_hybrid_combines_by_the_geometric_mean_unless_told_otherwise() -> None:
    """The geometric combination is the default: it is the one the solver maximizes."""
    # --- act / assert -----------------
    hybrid = DiversityObjectiveHybrid(
        (
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
        )
    )
    assert hybrid.combination == HybridCombination.GEOMETRIC_MEAN


# =================================================================================================
#  tracker_specs (and the facts derived from it)
# =================================================================================================
@pytest.mark.parametrize(
    "objective, expected_specs, expected_distinct_specs",
    [
        pytest.param(
            DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE),
            (DiversityTrackerSpec(None, MEAN_DISTANCE),),
            (DiversityTrackerSpec(None, MEAN_DISTANCE),),
            id="simple",
        ),
        pytest.param(
            _hybrid(
                DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE),  # (None, MEAN_DISTANCE)
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),  # (L1, SEPARATION)
                DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L1),  # repeat: (L1, SEPARATION)
            ),
            (
                DiversityTrackerSpec(None, MEAN_DISTANCE),
                DiversityTrackerSpec(L1, SEPARATION),
                DiversityTrackerSpec(L1, SEPARATION),
            ),
            (DiversityTrackerSpec(None, MEAN_DISTANCE), DiversityTrackerSpec(L1, SEPARATION)),
            id="hybrid_keeps_repeats_and_dedups_in_first_seen_order",
        ),
    ],
)
def test_tracker_specs(objective, expected_specs, expected_distinct_specs) -> None:
    """`tracker_specs` lists one spec per array `compute` takes; `distinct_tracker_specs` dedups them, first-seen."""
    # --- act / assert -----------------
    assert objective.tracker_specs == expected_specs
    assert objective.distinct_tracker_specs == expected_distinct_specs


def test_a_simple_objectives_tracker_spec_is_its_one_spec() -> None:
    """`tracker_spec` is the shorthand for the single entry of a simple objective's `tracker_specs`."""
    # --- act / assert -----------------
    objective = DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1)
    assert objective.tracker_spec == DiversityTrackerSpec(L1, SEPARATION)
    assert objective.tracker_specs == (objective.tracker_spec,)


@pytest.mark.parametrize(
    "objective, expected",
    [
        (DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION), (None,)),
        (
            _hybrid(
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
                DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L1),
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
        (
            _hybrid(
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
            ),
            False,  # two distinct specs
        ),
        (
            _hybrid(
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L1),
            ),
            True,  # two terms over one separation spec
        ),
    ],
)
def test_has_single_separation_tracker(objective, expected) -> None:
    """One distinct separation spec is the batched-init case; a second spec or another family is not."""
    # --- act / assert -----------------
    assert objective.has_single_separation_tracker() is expected


# =================================================================================================
#  compute
# =================================================================================================
def test_simple_computes_its_metric_over_its_one_spec() -> None:
    """A simple objective reduces its one spec's contribution array with its diversity metric."""
    # --- arrange ----------------------
    objective = DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION)
    contributions = (_f32([4.0, 2.0, 6.0]),)

    # --- act / assert -----------------
    assert objective.compute(contributions) == pytest.approx(2.0)  # min of the separation array


@pytest.mark.parametrize(
    "terms, combination, contributions, expected",
    [
        pytest.param(
            (
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
            ),
            HybridCombination.GEOMETRIC_MEAN,
            (_f32([4.0, 8.0]), _f32([9.0, 3.0])),  # L1 array (min 4), L2 array (min 3)
            np.sqrt(4.0 * 3.0),
            id="geomean_of_two_distances",
        ),
        pytest.param(
            (
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                DiversityObjectiveSimple(DiversityMetric.MEAN_SEPARATION, L1),  # same spec as the first term
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
            ),
            HybridCombination.GEOMETRIC_MEAN,
            (_f32([4.0, 8.0]), _f32([4.0, 8.0]), _f32([9.0, 3.0])),  # the L1 array twice (min 4, mean 6), L2 (min 3)
            (4.0 * 6.0 * 3.0) ** (1.0 / 3.0),
            id="geomean_with_a_shared_spec_receives_that_array_twice",
        ),
        pytest.param(
            (
                DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L1),
                DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L2),
            ),
            HybridCombination.ARITHMETIC_MEAN,
            (_f32([0.0, 0.0, 1.0, 2.0]), _f32([0.0, 3.0, 1.0, 2.0])),  # fractions 2/4 and 3/4
            (0.5 + 0.75) / 2,
            id="arithmetic_mean_of_two_fractions",
        ),
        pytest.param(
            (
                DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L1),
                DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L2),
            ),
            HybridCombination.ARITHMETIC_MEAN,
            (_f32([0.0, 0.0]), _f32([0.0, 3.0])),  # fractions 0 and 1/2: the mean is not pinned at zero
            0.25,
            id="arithmetic_mean_survives_an_all_zero_term",
        ),
    ],
)
def test_hybrid_computes_the_combination_of_its_terms(terms, combination, contributions, expected) -> None:
    """A hybrid returns the geometric or arithmetic mean of its terms' scores, one array per term."""
    # --- arrange ----------------------
    objective = DiversityObjectiveHybrid(terms, combination)

    # --- act / assert -----------------
    assert objective.compute(contributions) == pytest.approx(expected, rel=1e-5)  # float32 arithmetic


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
    assert tie_breakers == [DiversityObjectiveSimple(metric, L2) for metric in expected_tie_breaker_metrics]


def test_a_geometric_hybrids_default_tie_breakers_are_hybrids_over_its_distinct_distances() -> None:
    """The approximate geomean combines geometrically and the non-zero fraction arithmetically, over each distance."""
    # --- arrange ----------------------
    objective = _hybrid(
        DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
        DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L2),
        DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),  # a repeated distance counts once
    )

    # --- act / assert -----------------
    assert objective.default_tie_breakers() == [
        _hybrid(
            DiversityObjectiveSimple(DiversityMetric.APPROX_GEOMEAN_SEPARATION, L1),
            DiversityObjectiveSimple(DiversityMetric.APPROX_GEOMEAN_SEPARATION, L2),
            combination=HybridCombination.GEOMETRIC_MEAN,
        ),
        _hybrid(
            DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L1),
            DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L2),
            combination=HybridCombination.ARITHMETIC_MEAN,
        ),
    ]


def test_an_arithmetic_hybrid_has_no_tie_breakers() -> None:
    """An arithmetic hybrid is only ever a tie-breaker, and a tie-breaker is not ranked by further tie-breakers."""
    # --- act / assert -----------------
    objective = _hybrid(
        DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L1),
        DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, L2),
        combination=HybridCombination.ARITHMETIC_MEAN,
    )
    assert objective.default_tie_breakers() == []
