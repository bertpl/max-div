import numpy as np
import pytest

from max_div._core._math.geomean import geomean_f32
from max_div._core.metrics import (
    HybridAggregationArithmeticMean,
    HybridAggregationBase,
    HybridAggregationGeometricMean,
    HybridAggregationMinimum,
)


def _random_rows(n_rows: int, n_terms: int) -> np.ndarray:
    """Return a C-contiguous float32 matrix of positive values, one column per term."""
    return np.random.default_rng(42).uniform(0.001, 10.0, size=(n_rows, n_terms)).astype(np.float32)


# ==================================================================================================
#  Weights
# ==================================================================================================
@pytest.mark.parametrize("aggregation_type", HybridAggregationBase.__subclasses__())
def test_with_unit_weights_weights_every_term_1(aggregation_type) -> None:
    """An aggregation built with unit weights holds a weight of 1 per term and has `has_non_unit_weights` False."""
    # --- act --------------------------
    aggregation = aggregation_type.with_unit_weights(3)

    # --- assert -----------------------
    assert aggregation.weights == (1.0, 1.0, 1.0)
    assert not aggregation.has_non_unit_weights


def test_weights_are_stored_as_floats_and_compare_by_value() -> None:
    """Integer and numpy weights are stored as plain floats, and aggregations compare by type and weights."""
    # --- act --------------------------
    aggregation = HybridAggregationGeometricMean((2, np.float32(0.5)))

    # --- assert -----------------------
    assert aggregation.weights == (2.0, 0.5)
    assert all(type(weight) is float for weight in aggregation.weights)
    assert aggregation.has_non_unit_weights
    assert aggregation == HybridAggregationGeometricMean((2.0, 0.5))
    assert aggregation != HybridAggregationArithmeticMean((2.0, 0.5))


@pytest.mark.parametrize(
    "weight",
    [0.0, -2.0, float("inf"), float("nan"), True, "2", None],
    ids=["zero", "negative", "infinite", "nan", "bool", "string", "none"],
)
def test_a_weight_that_is_not_a_positive_finite_number_is_rejected(weight) -> None:
    """A zero, negative, infinite, NaN, bool or non-numeric weight raises ValueError."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="positive, finite number"):
        HybridAggregationArithmeticMean((1.0, weight))


def test_validate_term_count_rejects_a_weight_count_that_differs() -> None:
    """`validate_term_count` accepts the aggregation's own term count and rejects any other."""
    # --- arrange ----------------------
    aggregation = HybridAggregationGeometricMean.with_unit_weights(2)

    # --- act / assert -----------------
    aggregation.validate_term_count(2)
    with pytest.raises(ValueError, match="got 2 weights for 3 terms"):
        aggregation.validate_term_count(3)


@pytest.mark.parametrize(
    "aggregation, expected",
    [
        (HybridAggregationGeometricMean((1.0, 1.0)), "geomean(A, B)"),
        (HybridAggregationArithmeticMean((1.0, 1.0)), "mean(A, B)"),
        (HybridAggregationArithmeticMean((31.6227766, 1000.0)), "mean(A, B; weights 31.62, 1000)"),
        (HybridAggregationMinimum((31.6227766, 1000.0)), "min(A, B; weights 31.62, 1000)"),
    ],
)
def test_format_label_names_the_aggregation_and_the_weights_unless_all_1(aggregation, expected) -> None:
    """The label is the aggregation's name over the terms, with the weights appended only when they are not all 1."""
    # --- act / assert -----------------
    assert aggregation.format_label(["A", "B"]) == expected


# ==================================================================================================
#  Combination at equal weights: bit for bit `geomean_f32` and numpy's float32 mean
# ==================================================================================================
@pytest.mark.parametrize(
    "aggregation_type, row_mean",
    [
        (HybridAggregationGeometricMean, geomean_f32),
        (HybridAggregationArithmeticMean, lambda row: np.mean(row, dtype=np.float32)),
    ],
    ids=["geometric", "arithmetic"],
)
def test_an_aggregation_at_equal_weights_is_bit_for_bit_the_unweighted_mean(aggregation_type, row_mean) -> None:
    """At equal weights both the row combination and the score are exactly the unweighted mean of the row."""
    # --- arrange ----------------------
    rows = _random_rows(1000, 3)
    aggregation = aggregation_type.with_unit_weights(3)

    # --- act --------------------------
    aggregated_rows = aggregation.aggregate_rows(rows)
    score = aggregation.aggregate_scores(rows[0, :])

    # --- assert -----------------------
    np.testing.assert_array_equal(aggregated_rows, [row_mean(rows[i, :]) for i in range(1000)])
    assert score == float(row_mean(rows[0, :]))


# ==================================================================================================
#  Combination with weights
# ==================================================================================================
@pytest.mark.parametrize(
    "aggregation, expected",
    [
        (HybridAggregationGeometricMean((2.0, 1.0)), (np.array([4.0, 9.0]) ** 2 * np.array([2.0, 3.0])) ** (1.0 / 3.0)),
        (HybridAggregationArithmeticMean((3.0, 1.0)), (3.0 * np.array([4.0, 9.0]) + np.array([2.0, 3.0])) / 4.0),
        (HybridAggregationMinimum((1.0, 4.0)), np.minimum(np.array([4.0, 9.0]), 4.0 * np.array([2.0, 3.0]))),
    ],
    ids=["geometric", "arithmetic", "minimum"],
)
def test_a_weighted_aggregation_combines_each_row_and_the_scores_alike(aggregation, expected) -> None:
    """Each row combines by the weighted formula, and `aggregate_scores` over one row gives the same value."""
    # --- arrange ----------------------
    rows = np.array([[4.0, 2.0], [9.0, 3.0]], dtype=np.float32)

    # --- act --------------------------
    aggregated_rows = aggregation.aggregate_rows(rows)
    score = aggregation.aggregate_scores(rows[1, :])

    # --- assert -----------------------
    np.testing.assert_allclose(aggregated_rows, expected, rtol=1e-6)
    assert aggregated_rows.dtype == np.float32
    assert score == pytest.approx(expected[1], rel=1e-6)
