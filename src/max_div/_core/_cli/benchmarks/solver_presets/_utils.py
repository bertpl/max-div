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
    # --- per problem? ---------------------------
    if per_problem_execution:
        all_problem_names = {p.problem_name for p in params}
        return sum(
            [
                estimate_execution_time_sec_multi(
                    params=[p for p in params if p.problem_name == problem_name],
                    per_problem_execution=False,
                )
                for problem_name in all_problem_names
            ]
        )

    # --- split ----------------------------------
    # parallel runs each use all cores, so they execute one after another and cannot be packed
    # into the process pool the single-worker runs share
    single_params = [p for p in params if not p.is_parallel]
    parallel_sec = sum(estimate_execution_time_sec_single(p) for p in params if p.is_parallel)

    # --- estimate packed duration ---------------
    packed_sec = 0.0
    if single_params:
        durations_sec = [estimate_execution_time_sec_single(p) for p in single_params]
        n_processes = get_n_processes(len(single_params))
        packed_sec = max(*durations_sec, sum(durations_sec) / n_processes)

    return packed_sec + parallel_sec


def estimate_execution_time_sec_single(params: SolverPresetBenchmarkParams) -> float:
    """Estimate execution time in seconds for a single benchmark run.

    Runs carry an end-to-end budget, so setup (distance computation, worker spawning) is spent
    inside the budget rather than on top of it — but a budget cannot cut setup short, so setup
    is the estimate whenever it exceeds the budget.
    """
    setup_sec = 4.0 * ((params.problem_size / 10000.0) ** 2)  # initial distance computation is O(n^2)
    if params.is_parallel:
        setup_sec += 2.0  # parallel runs also spawn worker processes
    return max(params.duration.value(), setup_sec)


# =================================================================================================
#  Processor counts
# =================================================================================================
def get_n_processes(n_scope: int) -> int:
    """Determine appropriate number of processes for multiprocessing using scope size & core count."""
    return min(n_scope, round(0.75 * (os.cpu_count() or 1)))  # cpu_count() can return None
