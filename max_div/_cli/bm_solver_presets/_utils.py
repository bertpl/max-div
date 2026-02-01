import os

from ._models import SolverPresetBenchmarkParams


# =================================================================================================
#  Progress bars
# =================================================================================================
def get_pbar_units(params: SolverPresetBenchmarkParams) -> int:
    return max(1, round(estimate_execution_time_sec_single(params)))


# =================================================================================================
#  Estimate time durations
# =================================================================================================
def estimate_execution_time_sec_multi(params: list[SolverPresetBenchmarkParams]) -> float:
    """Estimate total execution time in seconds for multiple benchmark runs, taking multiprocessing into account."""

    # --- init ------------------------
    durations_sec = [estimate_execution_time_sec_single(p) for p in params]
    sum_duration_sec = sum(durations_sec)
    max_duration_sec = max(durations_sec)

    # --- estimate total duration -----
    n_processes = get_n_processes(len(params))
    return max(max_duration_sec, sum_duration_sec / n_processes)


def estimate_execution_time_sec_single(params: SolverPresetBenchmarkParams) -> float:
    """Estimate execution time in seconds for a single benchmark run."""
    fixed_overhead_sec = 1.0
    solver_init_overhead_sec = params.problem_size / 100.0
    return params.duration.value() + fixed_overhead_sec + solver_init_overhead_sec


# =================================================================================================
#  Processor counts
# =================================================================================================
def get_n_processes(n_scope: int) -> int:
    """Determine appropriate number of processes for multiprocessing using scope size & core count."""
    return min(n_scope, round(0.75 * os.cpu_count()))
