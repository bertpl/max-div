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
def estimate_execution_time_sec_multi(
    params: list[SolverPresetBenchmarkParams], per_problem_execution: bool = True
) -> float:
    """Estimate total execution time in seconds for multiple benchmark runs, taking multiprocessing into account."""

    # --- per problem? ----------------
    if per_problem_execution:
        all_problem_names = set(p.problem_name for p in params)
        return sum(
            [
                estimate_execution_time_sec_multi(
                    params=[p for p in params if p.problem_name == problem_name],
                    per_problem_execution=False,
                )
                for problem_name in all_problem_names
            ]
        )

    # --- init ------------------------
    durations_sec = [estimate_execution_time_sec_single(p) for p in params]
    sum_duration_sec = sum(durations_sec)
    max_duration_sec = max(durations_sec)

    # --- estimate total duration -----
    n_processes = get_n_processes(len(params))
    return max(max_duration_sec, sum_duration_sec / n_processes)


def estimate_execution_time_sec_single(params: SolverPresetBenchmarkParams) -> float:
    """Estimate execution time in seconds for a single benchmark run."""
    overhead_sec = 4.0 * ((params.problem_size / 100.0) ** 2)  # initial distance computation of solver init is O(n^2)
    return params.duration.value() + overhead_sec


# =================================================================================================
#  Processor counts
# =================================================================================================
def get_n_processes(n_scope: int) -> int:
    """Determine appropriate number of processes for multiprocessing using scope size & core count."""
    return min(n_scope, round(0.75 * os.cpu_count()))
