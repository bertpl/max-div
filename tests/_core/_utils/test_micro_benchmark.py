"""Pin `benchmark`'s control logic: adaptive `n_executions` sizing, baseline subtraction, quantile stats.

`benchmark` is a feedback controller -- each run rescales `n_executions` from the previous run's
`Timer` reading toward the per-run time target. Timing enters only through `time.perf_counter_ns`,
so most of these tests drive the `fake_clock` fixture, advanced only by the workload's own patched
`sleep`; every assertion is then exact and immune to a loaded runner. One test at the bottom
exercises the real clock, asserting only a floor, so faking cannot hide a benchmark that never times
anything.
"""

import time
from time import perf_counter_ns

import pytest

from max_div._core._utils import BenchmarkResult, benchmark


# =================================================================================================
#  Control logic (fake clock)
# =================================================================================================
@pytest.mark.parametrize("index_range", [None, 1000])
def test_benchmark_stats_recover_the_cost_per_execution(fake_clock, index_range: int | None):
    """Every run measures exactly the workload's cost, so all three quantiles equal it."""
    # --- arrange -----------------------------------------
    cost_sec = 1e-4

    def f_test(_idx: int = 0) -> None:
        time.sleep(cost_sec)  # patched by fake_clock: advances the clock by exactly cost_sec

    # --- act ---------------------------------------------
    result = benchmark(f_test, t_per_run=1e-2, n_warmup=5, n_benchmark=10, silent=True, index_range=index_range)

    # --- assert ------------------------------------------
    assert result.t_sec_q_25 == result.t_sec_q_50 == result.t_sec_q_75 == pytest.approx(cost_sec)


def test_benchmark_adapts_n_executions_upward(fake_clock):
    """With a per-run target well above the workload cost, the controller ramps past one call per run."""
    # --- arrange -----------------------------------------
    cost_sec = 1e-5  # target 1e-2 / cost 1e-5 -> converges around 1000 executions per run
    n_calls = 0

    def f_test(_idx: int = 0) -> None:
        nonlocal n_calls
        n_calls += 1
        time.sleep(cost_sec)

    n_warmup, n_benchmark = 5, 5

    # --- act ---------------------------------------------
    benchmark(f_test, t_per_run=1e-2, n_warmup=n_warmup, n_benchmark=n_benchmark, silent=True)

    # --- assert ------------------------------------------
    # per run the controller converges near t_per_run / cost = 1000 executions, so the total call
    # count clears that target -- a controller stuck at one call per run would be far below it
    assert n_calls > 1000


def test_benchmark_survives_a_run_measuring_zero_elapsed(fake_clock):
    """A run whose measured time rounds to zero must not raise -- t_tot is a divisor in the rescale."""
    # --- arrange -----------------------------------------
    costs = iter([0.0])  # the first (warm-up) run measures nothing; every run after it costs 1e-4

    def f_test(_idx: int = 0) -> None:
        time.sleep(next(costs, 1e-4))

    # --- act ---------------------------------------------
    result = benchmark(f_test, t_per_run=1e-2, n_warmup=3, n_benchmark=3, silent=True)

    # --- assert ------------------------------------------
    # not raising is the point; the benchmark runs all cost 1e-4, so the zero warm-up leaves no trace
    assert result.t_sec_q_50 == pytest.approx(1e-4)


def test_benchmark_prints_progress_unless_silent(fake_clock, capsys):
    """Non-silent mode prints the warmup/benchmark markers and the final per-execution figure; silent mode is mute."""

    # --- arrange -----------------------------------------
    def f_test(_idx: int = 0) -> None:
        time.sleep(1e-4)

    # --- act / assert: non-silent ------------------------
    benchmark(f_test, t_per_run=1e-2, n_warmup=2, n_benchmark=3, silent=False)
    printed = capsys.readouterr().out
    assert "Benchmarking" in printed
    assert "ww..." in printed  # 2 warm-up markers then 3 benchmark-run markers, in order
    assert "per execution" in printed  # the final t_sec_with_uncertainty_str line

    # --- act / assert: silent ----------------------------
    benchmark(f_test, t_per_run=1e-2, n_warmup=2, n_benchmark=3, silent=True)
    assert capsys.readouterr().out == ""


# =================================================================================================
#  Aggregation
# =================================================================================================
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


# =================================================================================================
#  Against the real clock
# =================================================================================================
def test_benchmark_measures_the_real_clock():
    """Guard against the fake clock hiding a benchmark that never times anything.

    Asserts only a floor: a busy-wait guarantees a minimum of real elapsed time, a loaded runner
    inflates it without limit, and the harness subtracts a baseline and divides by an execution
    count -- so half the busy-wait is the strongest bound that cannot flake.
    """
    # --- arrange -----------------------------------------
    t_sleep = 1e-4

    def f_test(_idx: int = 0) -> None:
        t_start = perf_counter_ns()  # the real clock, immune to fake_clock (not requested here)
        t_end = t_start + (1e9 * t_sleep)
        while perf_counter_ns() < t_end:
            pass

    # --- act ---------------------------------------------
    result = benchmark(f_test, t_per_run=1e-2, n_warmup=10, n_benchmark=20, silent=True)

    # --- assert ------------------------------------------
    assert result.t_sec_q_50 >= 0.5 * t_sleep
