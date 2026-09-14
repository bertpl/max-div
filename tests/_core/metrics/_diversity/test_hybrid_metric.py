import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityMetric,
    DiversityObjectiveHybrid,
    DiversityObjectiveSimple,
    DiversityTerm,
    HybridDiversityMetric,
    HybridObjectiveType,
)
from max_div._core.metrics._diversity._hybrid_metric import _AGGREGATION_LABELS

_AXIS_0 = DistanceMetric.along_axis(0)


def _two_term_hybrid(factory) -> HybridDiversityMetric:
    """Return a hybrid built by `factory` from a bare min-separation term and a geomean term along axis 0."""
    return factory(DiversityMetric.MIN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION.over(_AXIS_0))


# =================================================================================================
#  DiversityTerm
# =================================================================================================
def test_over_pairs_the_diversity_metric_with_the_distance_metric() -> None:
    # --- act --------------------------
    term = DiversityMetric.MIN_SEPARATION.over(_AXIS_0)

    # --- assert -----------------------
    assert isinstance(term, DiversityTerm)
    assert term.diversity_metric == DiversityMetric.MIN_SEPARATION
    assert term.distance_metric == _AXIS_0


def test_terms_compare_and_hash_by_value() -> None:
    # --- arrange ----------------------
    term = DiversityMetric.MIN_SEPARATION.over(_AXIS_0)
    same = DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(0))
    other_distance = DiversityMetric.MIN_SEPARATION.over(DistanceMetric.l2_euclidean())
    other_diversity = DiversityMetric.MEAN_SEPARATION.over(_AXIS_0)

    # --- assert -----------------------
    assert term == same
    assert hash(term) == hash(same)
    assert term != other_distance
    assert term != other_diversity
    assert term != DiversityMetric.MIN_SEPARATION


def test_a_terms_label_and_repr_name_both_metrics() -> None:
    # --- arrange ----------------------
    term = DiversityMetric.MIN_SEPARATION.over(_AXIS_0)

    # --- assert -----------------------
    assert term.label == "MIN_SEPARATION over axis 0"
    assert repr(term) == "DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(0))"


# =================================================================================================
#  HybridDiversityMetric
# =================================================================================================
@pytest.mark.parametrize(
    "factory, aggregation",
    [
        (HybridDiversityMetric.geomean_of, HybridObjectiveType.GEOMETRIC_MEAN),
        (HybridDiversityMetric.mean_of, HybridObjectiveType.ARITHMETIC_MEAN),
    ],
)
def test_a_hybrid_resolves_to_a_hybrid_objective_with_its_aggregation(factory, aggregation) -> None:
    # --- arrange ----------------------
    hybrid = _two_term_hybrid(factory)

    # --- act --------------------------
    objective = hybrid._to_objective()

    # --- assert -----------------------
    assert objective == DiversityObjectiveHybrid(
        (
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION),
            DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, _AXIS_0),
        ),
        aggregation,
    )


def test_a_hybrid_keeps_its_terms_as_given_and_lists_the_distinct_distance_metrics() -> None:
    # --- arrange ----------------------
    axis_term = DiversityMetric.MIN_SEPARATION.over(_AXIS_0)
    hybrid = HybridDiversityMetric.geomean_of(DiversityMetric.MIN_SEPARATION, axis_term, axis_term)

    # --- assert -----------------------
    assert hybrid.terms == (DiversityMetric.MIN_SEPARATION, axis_term, axis_term)
    assert hybrid.named_distance_metrics == (_AXIS_0,)


def test_a_repeated_term_counts_once_per_repeat() -> None:
    # --- arrange ----------------------
    axis_term = DiversityMetric.MIN_SEPARATION.over(_AXIS_0)

    # --- act --------------------------
    objective = HybridDiversityMetric.mean_of(DiversityMetric.MIN_SEPARATION, axis_term, axis_term)._to_objective()

    # --- assert -----------------------
    assert len(objective.terms) == 3


def test_every_aggregation_type_has_a_label() -> None:
    assert set(_AGGREGATION_LABELS) == set(HybridObjectiveType)


def test_a_hybrid_needs_at_least_two_terms() -> None:
    with pytest.raises(ValueError, match="at least two terms"):
        HybridDiversityMetric.geomean_of(DiversityMetric.MIN_SEPARATION)


def test_a_hybrid_does_not_nest() -> None:
    # --- arrange ----------------------
    inner = _two_term_hybrid(HybridDiversityMetric.geomean_of)

    # --- act / assert -----------------
    with pytest.raises(TypeError, match="got HybridDiversityMetric"):
        HybridDiversityMetric.mean_of(inner, DiversityMetric.MIN_SEPARATION)


def test_hybrids_compare_and_hash_by_terms_and_aggregation() -> None:
    # --- arrange ----------------------
    geomean = _two_term_hybrid(HybridDiversityMetric.geomean_of)
    same = _two_term_hybrid(HybridDiversityMetric.geomean_of)
    mean = _two_term_hybrid(HybridDiversityMetric.mean_of)

    # --- assert -----------------------
    assert geomean == same
    assert hash(geomean) == hash(same)
    assert geomean != mean
    assert geomean != DiversityMetric.MIN_SEPARATION


@pytest.mark.parametrize(
    "factory, expected_label, expected_repr",
    [
        (
            HybridDiversityMetric.geomean_of,
            "geomean(MIN_SEPARATION, GEOMEAN_SEPARATION over axis 0)",
            "HybridDiversityMetric.geomean_of(DiversityMetric.MIN_SEPARATION, "
            "DiversityMetric.GEOMEAN_SEPARATION.over(DistanceMetric.along_axis(0)))",
        ),
        (
            HybridDiversityMetric.mean_of,
            "mean(MIN_SEPARATION, GEOMEAN_SEPARATION over axis 0)",
            "HybridDiversityMetric.mean_of(DiversityMetric.MIN_SEPARATION, "
            "DiversityMetric.GEOMEAN_SEPARATION.over(DistanceMetric.along_axis(0)))",
        ),
    ],
)
def test_a_hybrids_label_and_repr_name_the_aggregation_and_the_terms(factory, expected_label, expected_repr) -> None:
    # --- arrange ----------------------
    hybrid = _two_term_hybrid(factory)

    # --- assert -----------------------
    assert hybrid.label == expected_label
    assert repr(hybrid) == expected_repr
