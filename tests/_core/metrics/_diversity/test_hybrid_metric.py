import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityMetric,
    DiversityObjectiveHybrid,
    DiversityObjectiveSimple,
    DiversityTerm,
    HybridAggregationArithmeticMean,
    HybridAggregationBase,
    HybridAggregationGeometricMean,
    HybridAggregationMinimum,
    HybridDiversityMetric,
)
from max_div._core.metrics._distance import FullMatrixDistanceSpec

_AXIS_0 = DistanceMetric.along_axis(0)
_SPEC_A = FullMatrixDistanceSpec(matrix_id=0, label="a")
_SPEC_B = FullMatrixDistanceSpec(matrix_id=1, label="b")


def _two_term_hybrid(factory, **kwargs) -> HybridDiversityMetric:
    """Return a hybrid built by `factory` from a bare min-separation term and a geomean term along axis 0."""
    return factory(DiversityMetric.MIN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION.over(_AXIS_0), **kwargs)


def _distance_spec_of(distance_metric: DistanceMetric | None) -> FullMatrixDistanceSpec:
    """Return spec `a` for a term that names no distance metric, and spec `b` for a term along axis 0."""
    return {None: _SPEC_A, _AXIS_0: _SPEC_B}[distance_metric]


# ==================================================================================================
#  DiversityTerm
# ==================================================================================================
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


@pytest.mark.parametrize(
    "term, expected_label, expected_repr",
    [
        (
            DiversityMetric.MIN_SEPARATION.over(_AXIS_0),
            "MIN_SEPARATION over axis 0",
            "DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(0))",
        ),
        (DiversityTerm(DiversityMetric.MIN_SEPARATION), "MIN_SEPARATION", "DiversityMetric.MIN_SEPARATION"),
    ],
    ids=["named_distance_metric", "no_distance_metric"],
)
def test_a_terms_label_and_repr_name_its_metrics(term: DiversityTerm, expected_label: str, expected_repr: str) -> None:
    """A term's label and repr name its diversity metric, and its distance metric when it names one."""
    # --- act / assert -----------------
    assert term.label == expected_label
    assert repr(term) == expected_repr


# ==================================================================================================
#  HybridDiversityMetric
# ==================================================================================================
@pytest.mark.parametrize(
    "factory, weights, aggregation",
    [
        (HybridDiversityMetric.geomean_of, None, HybridAggregationGeometricMean((1.0, 1.0))),
        (HybridDiversityMetric.mean_of, None, HybridAggregationArithmeticMean((1.0, 1.0))),
        (HybridDiversityMetric.geomean_of, (2, 0.5), HybridAggregationGeometricMean((2.0, 0.5))),
        (HybridDiversityMetric.mean_of, [3.0, 1.0], HybridAggregationArithmeticMean((3.0, 1.0))),
        (HybridDiversityMetric.min_of, None, HybridAggregationMinimum((1.0, 1.0))),
        (HybridDiversityMetric.min_of, (10, 100), HybridAggregationMinimum((10.0, 100.0))),
    ],
    ids=["geomean", "mean", "weighted_geomean", "weighted_mean", "min", "weighted_min"],
)
def test_a_hybrid_resolves_to_a_hybrid_objective_with_its_aggregation(factory, weights, aggregation) -> None:
    # --- arrange ----------------------
    hybrid = _two_term_hybrid(factory, weights=weights)

    # --- act --------------------------
    objective = hybrid._to_objective(_distance_spec_of)

    # --- assert -----------------------
    assert objective == DiversityObjectiveHybrid(
        (
            DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, _SPEC_A),
            DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, _SPEC_B),
        ),
        aggregation,
    )


def test_a_hybrid_asks_for_the_distance_spec_of_each_terms_distance_metric_none_included() -> None:
    """The hybrid passes each term's distance metric on unread, in term order, and `None` for a bare term."""
    # --- arrange ----------------------
    asked: list[DistanceMetric | None] = []

    def recording_distance_spec_of(distance_metric: DistanceMetric | None) -> FullMatrixDistanceSpec:
        """Record the distance metric that the hybrid asks about, and return spec `a`."""
        asked.append(distance_metric)
        return _SPEC_A

    # --- act --------------------------
    _two_term_hybrid(HybridDiversityMetric.geomean_of)._to_objective(recording_distance_spec_of)

    # --- assert -----------------------
    assert asked == [None, _AXIS_0]


def test_a_hybrid_holds_a_bare_metric_as_a_term_without_a_distance_metric_and_lists_the_named_ones() -> None:
    # --- arrange ----------------------
    axis_term = DiversityMetric.MIN_SEPARATION.over(_AXIS_0)
    hybrid = HybridDiversityMetric.geomean_of(DiversityMetric.MIN_SEPARATION, axis_term, axis_term)

    # --- assert -----------------------
    assert hybrid.terms == (DiversityTerm(DiversityMetric.MIN_SEPARATION), axis_term, axis_term)
    assert hybrid.named_distance_metrics == (_AXIS_0,)


def test_a_repeated_term_counts_once_per_repeat() -> None:
    # --- arrange ----------------------
    axis_term = DiversityMetric.MIN_SEPARATION.over(_AXIS_0)

    # --- act --------------------------
    objective = HybridDiversityMetric.mean_of(DiversityMetric.MIN_SEPARATION, axis_term, axis_term)._to_objective(
        _distance_spec_of
    )

    # --- assert -----------------------
    assert len(objective.terms) == 3


def test_every_aggregation_has_a_factory_named_after_it() -> None:
    """Each aggregation's `name` plus `_of` is the public factory that builds it, as `repr` assumes."""
    # --- arrange ----------------------
    aggregation_types = HybridAggregationBase.__subclasses__()

    # --- act --------------------------
    built = {
        aggregation_type: type(
            _two_term_hybrid(getattr(HybridDiversityMetric, f"{aggregation_type.name}_of"))
            ._to_objective(_distance_spec_of)
            .aggregation
        )
        for aggregation_type in aggregation_types
    }

    # --- assert -----------------------
    assert built == {aggregation_type: aggregation_type for aggregation_type in aggregation_types}


def test_a_hybrids_weights_are_one_per_term_and_1_unless_given() -> None:
    """A hybrid holds one weight per term: 1 when none are given, and the given values as floats otherwise."""
    # --- act / assert -----------------
    assert _two_term_hybrid(HybridDiversityMetric.geomean_of).weights == (1.0, 1.0)
    assert _two_term_hybrid(HybridDiversityMetric.mean_of, weights=(2, 3)).weights == (2.0, 3.0)


@pytest.mark.parametrize(
    "weights, message",
    [
        pytest.param((1.0,), "got 1 weights for 2 terms", id="too_few"),
        pytest.param((1.0, 1.0, 1.0), "got 3 weights for 2 terms", id="too_many"),
        pytest.param((1.0, 0.0), "positive, finite number; got 0.0", id="not_positive"),
    ],
)
def test_a_hybrid_rejects_weights_that_are_not_one_positive_finite_number_per_term(weights, message) -> None:
    """The factory rejects a weight count that differs from the term count, and an invalid weight."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match=message):
        _two_term_hybrid(HybridDiversityMetric.geomean_of, weights=weights)


def test_a_hybrid_needs_at_least_two_terms() -> None:
    with pytest.raises(ValueError, match="at least two terms"):
        HybridDiversityMetric.geomean_of(DiversityMetric.MIN_SEPARATION)


def test_a_hybrid_does_not_nest() -> None:
    # --- arrange ----------------------
    inner = _two_term_hybrid(HybridDiversityMetric.geomean_of)

    # --- act / assert -----------------
    with pytest.raises(TypeError, match="got HybridDiversityMetric"):
        HybridDiversityMetric.mean_of(inner, DiversityMetric.MIN_SEPARATION)


def test_hybrids_compare_and_hash_by_terms_aggregation_and_weights() -> None:
    """Hybrids are equal and hash alike when their terms, aggregation and weights are equal, int or float."""
    # --- arrange ----------------------
    geomean = _two_term_hybrid(HybridDiversityMetric.geomean_of)
    geomean_with_unit_weights = _two_term_hybrid(HybridDiversityMetric.geomean_of, weights=(1, 1))
    mean = _two_term_hybrid(HybridDiversityMetric.mean_of)
    weighted = _two_term_hybrid(HybridDiversityMetric.geomean_of, weights=(2, 1))

    # --- assert -----------------------
    assert geomean == geomean_with_unit_weights
    assert hash(geomean) == hash(geomean_with_unit_weights)
    assert geomean != mean
    assert geomean != weighted
    assert geomean != DiversityMetric.MIN_SEPARATION


@pytest.mark.parametrize(
    "factory, weights, expected_label, expected_repr",
    [
        (
            HybridDiversityMetric.geomean_of,
            None,
            "geomean(MIN_SEPARATION, GEOMEAN_SEPARATION over axis 0)",
            "HybridDiversityMetric.geomean_of(DiversityMetric.MIN_SEPARATION, "
            "DiversityMetric.GEOMEAN_SEPARATION.over(DistanceMetric.along_axis(0)))",
        ),
        (
            HybridDiversityMetric.mean_of,
            (1.0, 1.0),
            "mean(MIN_SEPARATION, GEOMEAN_SEPARATION over axis 0)",
            "HybridDiversityMetric.mean_of(DiversityMetric.MIN_SEPARATION, "
            "DiversityMetric.GEOMEAN_SEPARATION.over(DistanceMetric.along_axis(0)))",
        ),
        (
            HybridDiversityMetric.geomean_of,
            (31.6227766, 1000),
            "geomean(MIN_SEPARATION, GEOMEAN_SEPARATION over axis 0; weights 31.62, 1000)",
            "HybridDiversityMetric.geomean_of(DiversityMetric.MIN_SEPARATION, "
            "DiversityMetric.GEOMEAN_SEPARATION.over(DistanceMetric.along_axis(0)), weights=(31.6227766, 1000.0))",
        ),
        (
            HybridDiversityMetric.min_of,
            (31.6227766, 1000),
            "min(MIN_SEPARATION, GEOMEAN_SEPARATION over axis 0; weights 31.62, 1000)",
            "HybridDiversityMetric.min_of(DiversityMetric.MIN_SEPARATION, "
            "DiversityMetric.GEOMEAN_SEPARATION.over(DistanceMetric.along_axis(0)), weights=(31.6227766, 1000.0))",
        ),
    ],
    ids=["geomean", "mean_at_equal_weights", "weighted_geomean", "weighted_min"],
)
def test_a_hybrids_label_and_repr_name_the_aggregation_the_terms_and_any_weights(
    factory, weights, expected_label, expected_repr
) -> None:
    """A hybrid's label and repr name the aggregation and the terms, and the weights only when they are not all 1."""
    # --- arrange ----------------------
    hybrid = _two_term_hybrid(factory, weights=weights)

    # --- assert -----------------------
    assert hybrid.label == expected_label
    assert repr(hybrid) == expected_repr
