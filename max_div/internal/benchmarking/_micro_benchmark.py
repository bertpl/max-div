from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from max_div.internal.formatting import format_short_time_duration
from max_div.internal.utils import clip

from ._timer import Timer


# =================================================================================================
#  BenchmarkResult
# =================================================================================================
@dataclass(frozen=True)
class BenchmarkResult:
    t_sec_q_25: float
    t_sec_q_50: float
    t_sec_q_75: float

    @property
    def t_sec_str(self) -> str:
        s_median = format_short_time_duration(dt_sec=self.t_sec_q_50, right_aligned=True, spaced=True, long_units=True)
        return s_median

    @property
    def t_sec_with_uncertainty_str(self) -> str:
        s_median = self.t_sec_str
        s_perc = f"{50 * (self.t_sec_q_75 - self.t_sec_q_25) / self.t_sec_q_50:.1f}%"
        return f"{s_median} ± {s_perc}"

    @classmethod
    def aggregate(cls, results: list[BenchmarkResult], method: Literal["mean", "geomean", "sum"]) -> BenchmarkResult:
        """
        Aggregate multiple BenchmarkResult objects into a single result, by aggregating q25, q50, 75 values separately.

        :param results: List of BenchmarkResult objects to aggregate
        :param method: Aggregation method - "mean", "geomean" (geometric mean), or "sum"
        :return: Aggregated BenchmarkResult
        """
        if not results:
            raise ValueError("Cannot aggregate empty list of results")

        # Collect all quantile values
        q25_values = [r.t_sec_q_25 for r in results]
        q50_values = [r.t_sec_q_50 for r in results]
        q75_values = [r.t_sec_q_75 for r in results]

        # Apply the aggregation method
        if method == "mean":
            agg_q25 = np.mean(q25_values)
            agg_q50 = np.mean(q50_values)
            agg_q75 = np.mean(q75_values)
        elif method == "geomean":
            agg_q25 = np.exp(np.mean(np.log(q25_values)))
            agg_q50 = np.exp(np.mean(np.log(q50_values)))
            agg_q75 = np.exp(np.mean(np.log(q75_values)))
        elif method == "sum":
            agg_q25 = np.sum(q25_values)
            agg_q50 = np.sum(q50_values)
            agg_q75 = np.sum(q75_values)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return BenchmarkResult(t_sec_q_25=agg_q25, t_sec_q_50=agg_q50, t_sec_q_75=agg_q75)


# =================================================================================================
#  Main benchmarking function
# =================================================================================================
def benchmark(
    f: Callable,
    t_per_run: float = 0.1,
    n_warmup: int = 5,
    n_benchmark: int = 30,
    silent: bool = False,
) -> BenchmarkResult:
    """
    Adaptive micro-benchmarking function, to determine the duration/execution of the provided callable `f`.

    :param f: (Callable) Function to benchmark. Should take no arguments.
    :param t_per_run: (float, default=0.1) time in seconds we want to target per benchmarking run.
                      # of executions/run is adjusted to meet this target.
    :param n_warmup: (int, default=5) Number of warmup runs to perform before benchmarking.
    :param n_benchmark: (int, default=30) Number of benchmark runs to perform.
    :param silent: (bool, default=False) If True, suppresses any output during benchmarking.
    :return: Median estimate of duration/execution of `f` in seconds.
    """

    # --- init --------------------------------------------
    lst_t = []  # list of measured times per execution in seconds
    n_executions = 1  # number of executions per run, adjusted dynamically
    f_baseline = _baseline_fun  # baseline function to subtract overhead

    if not silent:
        print("Benchmarking: ", end="")

    # --- main loop ---------------------------------------
    for i in range(n_warmup + n_benchmark):
        # run
        with Timer() as timer_tot:
            # baseline
            with Timer() as timer_baseline:
                for _ in range(n_executions):
                    f_baseline()
            t_baseline = timer_baseline.t_elapsed_sec()

            # actual function
            with Timer() as timer_f:
                for _ in range(n_executions):
                    f()
            t_f = timer_f.t_elapsed_sec()

        # store results of benchmark runs
        if i >= n_warmup:
            lst_t.append(abs(t_f - t_baseline) / n_executions)  # abs value to avoid negative times for very fast 'f'.
            if not silent:
                print(".", end="")
        else:
            if not silent:
                print("w", end="")

        # adjust n_executions to bring t_tot closer to t_per_run
        # NOTE: during warmup we adjust n_executions at a log-scale to reach t_per_run target at end of warmup
        t_tot = timer_tot.t_elapsed_sec()
        n_executions = round(
            clip(
                value=n_executions * (t_per_run / t_tot) ** min(1.0, (i + 1) / n_warmup),
                min_value=max(1.0, n_executions / 100),
                max_value=n_executions * 100,
            )
        )

    # --- finalize ----------------------------------------
    q25, q50, q75 = np.percentile(lst_t, [25, 50, 75])
    result = BenchmarkResult(t_sec_q_25=q25, t_sec_q_50=q50, t_sec_q_75=q75)
    if not silent:
        print(f"   {result.t_sec_with_uncertainty_str} per execution")

    # --- return result -----------------------------------
    return result


# =================================================================================================
#  Baseline benchmarks
# =================================================================================================
def _baseline_fun():
    pass
