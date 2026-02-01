import pytest

from max_div.internal.markdown import TableAggregationType


@pytest.mark.parametrize(
    "agg_type, expected_result",
    [
        (TableAggregationType.MEAN, 7 / 3),
        (TableAggregationType.SUM, 7),
        (TableAggregationType.GEOMEAN, 2),
    ],
)
def test_table_aggregation_type(agg_type: TableAggregationType, expected_result: float):
    # --- arrange -----------------------------------------
    values = [1, 2, 4]

    # --- act ---------------------------------------------
    result = agg_type.aggregate_values(values)

    # --- assert ------------------------------------------
    assert result == pytest.approx(expected_result)
