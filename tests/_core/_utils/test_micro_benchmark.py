from time import perf_counter_ns

import pytest

from max_div._core._utils import BenchmarkResult, benchmark


@pytest.mark.parametrize("t_sleep", [1e-5, 1e-4, 1e-3])
@pytest.mark.parametrize("silent", [True, False])
@pytest.mark.parametrize("index_range", [None, 1000])
def test_micro_benchmark(t_sleep: float, silent: bool, index_range: int | None):
    """A busy-wait of a known duration measures on the order of that duration.

    Only a lower bound is asserted, and at half the busy-wait rather than its full length: the
    harness subtracts a baseline and divides by an execution count, which can shave the reported
    figure below `t_sleep` for the tiniest workloads, so half leaves margin for that. No upper
    bound — the harness measures real wall-clock, which a loaded runner inflates without limit, so
    any ceiling would eventually flake.
    """

    # --- arrange -----------------------------------------
    def f_test(_idx: int = 0):
        t_start = perf_counter_ns()
        t_end = t_start + (1e9 * t_sleep)
        while perf_counter_ns() < t_end:
            pass

    # --- act ---------------------------------------------
    result = benchmark(
        f_test,
        t_per_run=1e-2,
        n_warmup=10,
        n_benchmark=20,
        silent=silent,
        index_range=index_range,
    )

    # --- assert ------------------------------------------
    assert isinstance(result, BenchmarkResult)
    assert result.t_sec_q_25 <= result.t_sec_q_50 <= result.t_sec_q_75
    assert result.t_sec_q_50 >= 0.5 * t_sleep


@pytest.mark.parametrize(
    "method,expected_q25,expected_q50,expected_q75",
    [
        ("mean", 7 / 3, 14 / 3, 21 / 3),
        ("geomean", 2.0, 4.0, 6.0),
        ("sum", 7.0, 14.0, 21.0),
    ],
)
def test_micro_benchmark_result_aggregation(method, expected_q25, expected_q50, expected_q75):
    # --- arrange -----------------------------------------
    results = [
        BenchmarkResult(t_sec_q_25=1.0, t_sec_q_50=2.0, t_sec_q_75=3.0),
        BenchmarkResult(t_sec_q_25=2.0, t_sec_q_50=4.0, t_sec_q_75=6.0),
        BenchmarkResult(t_sec_q_25=4.0, t_sec_q_50=8.0, t_sec_q_75=12.0),
    ]

    # --- act ---------------------------------------------
    aggregated = BenchmarkResult.aggregate(results, method=method)

    # --- assert ------------------------------------------
    assert isinstance(aggregated, BenchmarkResult)
    assert aggregated.t_sec_q_25 == pytest.approx(expected_q25)
    assert aggregated.t_sec_q_50 == pytest.approx(expected_q50)
    assert aggregated.t_sec_q_75 == pytest.approx(expected_q75)


@pytest.mark.parametrize(
    "results,method",
    [
        ([], "mean"),
        ([BenchmarkResult(t_sec_q_25=1.0, t_sec_q_50=2.0, t_sec_q_75=3.0)], "invalid"),
    ],
)
def test_micro_benchmark_result_aggregation_raises_value_error(results, method):
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        BenchmarkResult.aggregate(results, method=method)
