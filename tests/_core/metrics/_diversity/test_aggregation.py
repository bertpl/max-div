import numpy as np
import pytest

from max_div._core._math.geomean import geomean_f32
from max_div._core.metrics import ArithmeticMeanAggregation, GeometricMeanAggregation


def _random_rows(n_rows: int, n_terms: int) -> np.ndarray:
    """Return a C-contiguous float32 matrix of positive values, one column per term."""
    return np.random.default_rng(42).uniform(0.001, 10.0, size=(n_rows, n_terms)).astype(np.float32)


# =================================================================================================
#  Weights
# =================================================================================================
@pytest.mark.parametrize("aggregation_type", [GeometricMeanAggregation, ArithmeticMeanAggregation])
def test_with_equal_weights_weights_every_term_1(aggregation_type) -> None:
    # --- act --------------------------
    aggregation = aggregation_type.with_equal_weights(3)

    # --- assert -----------------------
    assert aggregation.weights == (1.0, 1.0, 1.0)
    assert not aggregation.is_weighted


def test_weights_are_stored_as_floats_and_compare_by_value() -> None:
    """Integer and numpy weights become floats, so an aggregation equals the one given the same values as floats."""
    # --- act --------------------------
    aggregation = GeometricMeanAggregation((2, np.float32(0.5)))

    # --- assert -----------------------
    assert aggregation.weights == (2.0, 0.5)
    assert all(type(weight) is float for weight in aggregation.weights)
    assert aggregation.is_weighted
    assert aggregation == GeometricMeanAggregation((2.0, 0.5))
    assert aggregation != ArithmeticMeanAggregation((2.0, 0.5))


@pytest.mark.parametrize(
    "weight",
    [0.0, -2.0, float("inf"), float("nan"), True, "2", None],
    ids=["zero", "negative", "infinite", "nan", "bool", "string", "none"],
)
def test_a_weight_that_is_not_a_positive_finite_number_is_rejected(weight) -> None:
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="positive, finite number"):
        ArithmeticMeanAggregation((1.0, weight))


def test_check_term_count_rejects_a_weight_count_that_differs() -> None:
    # --- arrange ----------------------
    aggregation = GeometricMeanAggregation.with_equal_weights(2)

    # --- act / assert -----------------
    aggregation.check_term_count(2)
    with pytest.raises(ValueError, match="got 2 weights for 3 terms"):
        aggregation.check_term_count(3)


@pytest.mark.parametrize(
    "aggregation, expected",
    [
        (GeometricMeanAggregation((1.0, 1.0)), "geomean(A, B)"),
        (ArithmeticMeanAggregation((1.0, 1.0)), "mean(A, B)"),
        (ArithmeticMeanAggregation((31.6227766, 1000.0)), "mean(A, B; weights 31.62, 1000)"),
    ],
)
def test_format_label_names_the_aggregation_and_the_weights_unless_all_1(aggregation, expected) -> None:
    # --- act / assert -----------------
    assert aggregation.format_label(["A", "B"]) == expected


# =================================================================================================
#  Combination at equal weights: bit for bit today's unweighted means
# =================================================================================================
def test_the_geometric_mean_at_equal_weights_is_bit_for_bit_the_unweighted_one() -> None:
    """At equal weights both the row combination and the score are exactly `geomean_f32` of the row."""
    # --- arrange ----------------------
    rows = _random_rows(1000, 3)
    aggregation = GeometricMeanAggregation.with_equal_weights(3)

    # --- act --------------------------
    aggregated_rows = aggregation.aggregate_rows(rows)
    score = aggregation.aggregate_scores(rows[0, :])

    # --- assert -----------------------
    np.testing.assert_array_equal(aggregated_rows, [geomean_f32(rows[i, :]) for i in range(1000)])
    assert score == float(geomean_f32(rows[0, :]))


def test_the_arithmetic_mean_at_equal_weights_is_bit_for_bit_numpys_float32_mean() -> None:
    """At equal weights the row combination is numpy's float32 row mean, and the score numpy's mean of the scores."""
    # --- arrange ----------------------
    rows = _random_rows(1000, 3)
    aggregation = ArithmeticMeanAggregation.with_equal_weights(3)

    # --- act --------------------------
    aggregated_rows = aggregation.aggregate_rows(rows)
    score = aggregation.aggregate_scores(rows[0, :])

    # --- assert -----------------------
    np.testing.assert_array_equal(aggregated_rows, rows.mean(axis=1, dtype=np.float32))
    assert score == float(np.mean(rows[0, :]))


# =================================================================================================
#  Combination with weights
# =================================================================================================
@pytest.mark.parametrize(
    "aggregation, expected",
    [
        (GeometricMeanAggregation((2.0, 1.0)), (np.array([4.0, 9.0]) ** 2 * np.array([2.0, 3.0])) ** (1.0 / 3.0)),
        (ArithmeticMeanAggregation((3.0, 1.0)), (3.0 * np.array([4.0, 9.0]) + np.array([2.0, 3.0])) / 4.0),
    ],
    ids=["geometric", "arithmetic"],
)
def test_a_weighted_aggregation_combines_each_row_and_the_scores_alike(aggregation, expected) -> None:
    """Each row combines by the aggregation's weighted mean; the score is that mean over the one row of scores."""
    # --- arrange ----------------------
    rows = np.array([[4.0, 2.0], [9.0, 3.0]], dtype=np.float32)

    # --- act --------------------------
    aggregated_rows = aggregation.aggregate_rows(rows)
    score = aggregation.aggregate_scores(rows[1, :])

    # --- assert -----------------------
    np.testing.assert_allclose(aggregated_rows, expected, rtol=1e-6)
    assert aggregated_rows.dtype == np.float32
    assert score == pytest.approx(expected[1], rel=1e-6)
