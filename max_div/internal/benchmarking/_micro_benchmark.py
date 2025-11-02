from dataclasses import dataclass
from typing import Callable

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


# =================================================================================================
#  Main benchmarking function
# =================================================================================================
def benchmark(
    f: Callable,
    t_per_run: float = 0.1,
    n_warmup: int = 10,
    n_benchmark: int = 30,
    silent: bool = False,
) -> BenchmarkResult:
    """
    Adaptive micro-benchmarking function, to determine the duration/execution of the provided callable `f`.

    :param f: (Callable) Function to benchmark. Should take no arguments.
    :param t_per_run: (float, default=0.1) time in seconds we want to target per benchmarking run.
                      # of executions/run is adjusted to meet this target.
    :param n_warmup: (int, default=10) Number of warmup runs to perform before benchmarking.
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

        # adjust n_executions
        t_tot = timer_tot.t_elapsed_sec()
        n_executions = round(
            clip(
                value=n_executions * (t_per_run / t_tot),
                min_value=max(1.0, n_executions / 10),
                max_value=n_executions * 10,
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
