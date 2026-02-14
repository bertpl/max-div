import pytest

from max_div.internal.benchmarking import BenchmarkResult
from max_div.internal.markdown import (
    TableAggregationType,
    TableElement,
    TablePercentage,
    TableText,
    TableTimeElapsed,
    TableValueRange,
    TableValueWithUncertainty,
)


# =================================================================================================
#  TableText
# =================================================================================================
def test_table_text():
    # --- act ---------------------------------------------
    table_text = TableText("my text\n**on two lines**")

    # --- assert ------------------------------------------
    assert isinstance(table_text, TableText)
    assert not table_text.supports_aggregation

    assert table_text.txt == ["my text", "**on two lines**"]
    assert table_text.to_plain_text() == ["my text", "on two lines"]
    assert table_text.to_mark_down() == "my text<br>**on two lines**"


def test_table_text_aggregate_not_implemented():
    # --- act & assert ------------------------------------
    with pytest.raises(NotImplementedError):
        TableText.aggregate(
            elements=[TableText("a"), TableText("b")],
            agg_type=TableAggregationType.SUM,
        )


def test_table_text_lt_and_equalish():
    # --- arrange -----------------------------------------
    text_a = TableText("apple")
    text_b = TableText("banana")
    text_c = TableText("Apple")

    # --- assert ------------------------------------------
    assert text_a < text_b
    assert not (text_b < text_a)

    assert text_a.is_equalish(text_c)
    assert not text_a.is_equalish(text_b)


# =================================================================================================
#  TablePercentage
# =================================================================================================
def test_table_percentage():
    # --- act ---------------------------------------------
    table_perc = TablePercentage(frac=1.23456789, decimals=2)

    # --- assert ------------------------------------------
    assert isinstance(table_perc, TablePercentage)
    assert table_perc.supports_aggregation

    assert table_perc.frac == 1.23456789
    assert table_perc.decimals == 2
    assert table_perc.to_plain_text() == ["123.46%"]
    assert table_perc.to_mark_down() == "123.46%"


def test_table_percentage_aggregate():
    # --- arrange -----------------------------------------
    perc_1 = TablePercentage(frac=0.1, decimals=1)
    perc_2 = TablePercentage(frac=0.2, decimals=1)
    perc_3 = TablePercentage(frac=0.4, decimals=1)
    perc_4 = TablePercentage(frac=0.9, decimals=1)

    # --- act ---------------------------------------------
    mean_perc = TablePercentage.aggregate(
        elements=[perc_1, perc_2, perc_3, perc_4],
        agg_type=TableAggregationType.MEAN,
    )

    # --- assert ------------------------------------------
    assert isinstance(mean_perc, TablePercentage)
    assert mean_perc.frac == pytest.approx(0.4)
    assert mean_perc.decimals == 2  # max decimals + 1


def test_table_percentage_lt_and_equalish():
    # --- arrange -----------------------------------------
    perc_1 = TablePercentage(frac=0.10, decimals=1)
    perc_2 = TablePercentage(frac=0.1001, decimals=1)
    perc_3 = TablePercentage(frac=0.40, decimals=1)

    # --- act & assert ------------------------------------
    assert perc_1 < perc_2 < perc_3
    assert min([perc_1, perc_2, perc_3]).frac == perc_1.frac

    assert perc_1.is_equalish(perc_2)
    assert perc_2.is_equalish(perc_1)

    assert not perc_1.is_equalish(perc_3)
    assert not perc_3.is_equalish(perc_1)


# =================================================================================================
#  TableTimeElapsed
# =================================================================================================
def test_table_time_elapsed():
    # --- act ---------------------------------------------
    table_time = TableTimeElapsed(t_sec_q_25=0.9, t_sec_q_50=1.0, t_sec_q_75=1.1)

    # --- assert ------------------------------------------
    assert isinstance(table_time, TableTimeElapsed)
    assert table_time.supports_aggregation

    assert table_time.q_25 == 0.9
    assert table_time.q_50 == 1.0
    assert table_time.q_75 == 1.1
    assert table_time.to_plain_text() == ["1.000 sec ± 10.0%"]
    assert table_time.to_mark_down() == "1.000 sec ± 10.0%"


def test_table_time_elapsed_aggregate():
    # --- arrange -----------------------------------------
    time_1 = TableTimeElapsed(t_sec_q_25=0.8, t_sec_q_50=1.0, t_sec_q_75=1.2)
    time_2 = TableTimeElapsed(t_sec_q_25=0.9, t_sec_q_50=1.1, t_sec_q_75=1.3)
    time_3 = TableTimeElapsed(t_sec_q_25=1.3, t_sec_q_50=1.5, t_sec_q_75=1.7)

    # --- act ---------------------------------------------
    mean_time = TableTimeElapsed.aggregate(
        elements=[time_1, time_2, time_3],
        agg_type=TableAggregationType.MEAN,
    )

    # --- assert ------------------------------------------
    assert isinstance(mean_time, TableTimeElapsed)
    assert mean_time.q_25 == pytest.approx(1.0)
    assert mean_time.q_50 == pytest.approx(1.2)
    assert mean_time.q_75 == pytest.approx(1.4)


def test_table_time_elapsed_factory_methods():
    # --- act ---------------------------------------------
    table_time_1 = TableTimeElapsed.from_benchmark_result(
        BenchmarkResult(
            t_sec_q_25=0.5,
            t_sec_q_50=1.0,
            t_sec_q_75=1.4,
        )
    )
    table_time_2 = TableTimeElapsed.from_values([0.8, 0.9, 1.0, 1.1, 1.2])

    # --- assert ------------------------------------------
    assert isinstance(table_time_1, TableTimeElapsed)
    assert table_time_1.q_25 == 0.5
    assert table_time_1.q_50 == 1.0
    assert table_time_1.q_75 == 1.4

    assert isinstance(table_time_2, TableTimeElapsed)
    assert table_time_2.q_25 == pytest.approx(0.9)
    assert table_time_2.q_50 == pytest.approx(1.0)
    assert table_time_2.q_75 == pytest.approx(1.1)


def test_table_time_elapsed_lt_and_equalish():
    # --- arrange -----------------------------------------
    # 4 times set up such that...
    #   - q50 increases strictly
    #   - time_1.q50 is in q25-q75 range of time_2, but NOT vice versa
    #   - time_3 & time_4 have their medians in each other's 25-75 percentile range
    time_1 = TableTimeElapsed(0.8, 0.91, 0.99)
    time_2 = TableTimeElapsed(0.9, 1.0, 1.1)
    time_3 = TableTimeElapsed(1.1, 1.2, 1.3)
    time_4 = TableTimeElapsed(1.11, 1.21, 1.31)

    # --- act & assert ------------------------------------
    assert time_1 < time_2 < time_3 < time_4

    assert not time_1.is_equalish(time_2)
    assert not time_2.is_equalish(time_1)

    assert not time_2.is_equalish(time_3)
    assert not time_3.is_equalish(time_2)

    assert time_3.is_equalish(time_4)
    assert time_4.is_equalish(time_3)


# =================================================================================================
#  TableValueWithUncertainty
# =================================================================================================
def test_table_value_with_uncertainty():
    # --- act ---------------------------------------------
    table_value = TableValueWithUncertainty(value_q_25=0.9, value_q_50=1.0, value_q_75=1.1, decimals=2)

    # --- assert ------------------------------------------
    assert isinstance(table_value, TableValueWithUncertainty)
    assert table_value.supports_aggregation

    assert table_value.q_25 == 0.9
    assert table_value.q_50 == 1.0
    assert table_value.q_75 == 1.1
    assert table_value.decimals == 2
    assert table_value.to_plain_text() == ["1.00 ± 10.0%"]
    assert table_value.to_mark_down() == "1.00 ± 10.0%"


def test_table_value_with_uncertainty_aggregate():
    # --- arrange -----------------------------------------
    value_1 = TableValueWithUncertainty(value_q_25=0.8, value_q_50=1.0, value_q_75=1.2, decimals=2)
    value_2 = TableValueWithUncertainty(value_q_25=0.9, value_q_50=1.1, value_q_75=1.3, decimals=2)
    value_3 = TableValueWithUncertainty(value_q_25=1.3, value_q_50=1.5, value_q_75=1.7, decimals=2)

    # --- act ---------------------------------------------
    mean_value = TableValueWithUncertainty.aggregate(
        elements=[value_1, value_2, value_3],
        agg_type=TableAggregationType.MEAN,
    )

    # --- assert ------------------------------------------
    assert isinstance(mean_value, TableValueWithUncertainty)
    assert mean_value.q_25 == pytest.approx(1.0)
    assert mean_value.q_50 == pytest.approx(1.2)
    assert mean_value.q_75 == pytest.approx(1.4)
    assert mean_value.decimals == 2  # max decimals


def test_table_value_with_uncertainty_from_values():
    # --- act ---------------------------------------------
    table_value = TableValueWithUncertainty.from_values([0.8, 0.9, 1.0, 1.1, 1.2], decimals=3)

    # --- assert ------------------------------------------
    assert isinstance(table_value, TableValueWithUncertainty)
    assert table_value.q_25 == pytest.approx(0.9)
    assert table_value.q_50 == pytest.approx(1.0)
    assert table_value.q_75 == pytest.approx(1.1)
    assert table_value.decimals == 3


def test_table_value_with_uncertainty_lt_and_equalish():
    # --- arrange -----------------------------------------
    # 4 values set up such that...
    #   - q50 increases strictly
    #   - value_1.q50 is in q25-q75 range of value_2, but NOT vice versa
    #   - value_3 & value_4 have their medians in each other's 25-75 percentile range
    value_1 = TableValueWithUncertainty(value_q_25=0.8, value_q_50=0.91, value_q_75=0.99, decimals=2)
    value_2 = TableValueWithUncertainty(value_q_25=0.9, value_q_50=1.0, value_q_75=1.1, decimals=2)
    value_3 = TableValueWithUncertainty(value_q_25=1.1, value_q_50=1.2, value_q_75=1.3, decimals=2)
    value_4 = TableValueWithUncertainty(value_q_25=1.11, value_q_50=1.21, value_q_75=1.31, decimals=2)

    # --- act & assert ------------------------------------
    assert value_1 < value_2 < value_3 < value_4

    assert not value_1.is_equalish(value_2)
    assert not value_2.is_equalish(value_1)

    assert not value_2.is_equalish(value_3)
    assert not value_3.is_equalish(value_2)

    assert value_3.is_equalish(value_4)
    assert value_4.is_equalish(value_3)


# =================================================================================================
#  TableValueRange
# =================================================================================================
@pytest.mark.parametrize(
    "q25, q50, q75, expected_str",
    [
        (0.8, 1.0, 1.2, "0.80...1.20"),
        (0.81111, 1.2222, 1.3333, "0.81...1.33"),
        (1234.56789, 1235.6789, 1236.789, "123(4.57...6.79)"),
        (123456, 234567, 345678, "123456...345678"),
        (888, 999, 1111, "888...1111"),
        (9.123, 10.345, 11.777, "9.1...11.8"),
        (111.111, 222.222, 333.333, "111...333"),
        (123.45678, 123.45678, 123.45678, "123.456780000"),
        # NOTE: next 3 cases are unexpected, since they're not sorted
        (1111, 999, 888, "1111...888"),
        (11.777, 10.345, 9.123, "11.8...9.1"),
        (333.333, 222.222, 111.111, "333...111"),
    ],
)
def test_table_value_range(q25: float, q50: float, q75: float, expected_str: str):
    # --- act ---------------------------------------------
    table_value = TableValueRange(value_q_25=q25, value_q_50=q50, value_q_75=q75, diff_decimals=3, max_decimals=9)

    # --- assert ------------------------------------------
    assert isinstance(table_value, TableValueRange)
    assert table_value.supports_aggregation

    assert table_value.q_25 == q25
    assert table_value.q_50 == q50
    assert table_value.q_75 == q75
    assert table_value.diff_decimals == 3
    assert table_value.to_plain_text() == [expected_str]
    assert table_value.to_mark_down() == expected_str


def test_table_value_range_aggregate():
    # --- arrange -----------------------------------------
    value_1 = TableValueRange(value_q_25=0.8, value_q_50=1.0, value_q_75=1.2, max_decimals=3, diff_decimals=2)
    value_2 = TableValueRange(value_q_25=0.9, value_q_50=1.1, value_q_75=1.3, max_decimals=3, diff_decimals=2)
    value_3 = TableValueRange(value_q_25=1.3, value_q_50=1.5, value_q_75=1.7, max_decimals=3, diff_decimals=2)

    # --- act ---------------------------------------------
    mean_value = TableValueRange.aggregate(
        elements=[value_1, value_2, value_3],
        agg_type=TableAggregationType.MEAN,
    )

    # --- assert ------------------------------------------
    assert isinstance(mean_value, TableValueRange)
    assert mean_value.q_25 == pytest.approx(1.0)
    assert mean_value.q_50 == pytest.approx(1.2)
    assert mean_value.q_75 == pytest.approx(1.4)
    assert mean_value.max_decimals == 3  # max of max_decimals
    assert mean_value.diff_decimals == 2  # max of diff_decimals


def test_table_value_range_from_values():
    # --- act ---------------------------------------------
    table_value = TableValueRange.from_values([0.8, 0.9, 1.0, 1.1, 1.2], max_decimals=5, diff_decimals=3)

    # --- assert ------------------------------------------
    assert isinstance(table_value, TableValueRange)
    assert table_value.q_25 == pytest.approx(0.9)
    assert table_value.q_50 == pytest.approx(1.0)
    assert table_value.q_75 == pytest.approx(1.1)
    assert table_value.max_decimals == 5
    assert table_value.diff_decimals == 3


def test_table_value_range_lt_and_equalish():
    # --- arrange -----------------------------------------
    # 4 values set up such that...
    #   - q50 increases strictly
    #   - value_1.q50 is in q25-q75 range of value_2, but NOT vice versa
    #   - value_3 & value_4 have their medians in each other's 25-75 percentile range
    value_1 = TableValueRange(value_q_25=0.8, value_q_50=0.91, value_q_75=0.99, max_decimals=3, diff_decimals=2)
    value_2 = TableValueRange(value_q_25=0.9, value_q_50=1.0, value_q_75=1.1, max_decimals=3, diff_decimals=2)
    value_3 = TableValueRange(value_q_25=1.1, value_q_50=1.2, value_q_75=1.3, max_decimals=3, diff_decimals=2)
    value_4 = TableValueRange(value_q_25=1.11, value_q_50=1.21, value_q_75=1.31, max_decimals=3, diff_decimals=2)

    # --- act & assert ------------------------------------
    assert value_1 < value_2 < value_3 < value_4

    assert not value_1.is_equalish(value_2)
    assert not value_2.is_equalish(value_1)

    assert not value_2.is_equalish(value_3)
    assert not value_3.is_equalish(value_2)

    assert value_3.is_equalish(value_4)
    assert value_4.is_equalish(value_3)
