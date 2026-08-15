import datetime
import multiprocessing
import os

from tqdm import tqdm

from max_div._core._timing import measure_end_to_end
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.solver import MaxDivSolverBuilder, ParallelMaxDivSolverBuilder, Verbosity

from ._models import SolverPresetBenchmarkExecutionInfo, SolverPresetBenchmarkParams, SolverPresetBenchmarkResult
from ._utils import get_pbar_units


# =================================================================================================
#  Execute MULTIPLE runs
# =================================================================================================
def executor_multi_parallel(
    scope: list[SolverPresetBenchmarkParams], n_processes: int
) -> list[SolverPresetBenchmarkResult]:
    # --- init --------------------------------------------
    n_pbar_units = sum([get_pbar_units(params) for params in scope])
    problem_names = sorted({params.problem_name for params in scope})
    desc = (
        "Executing preset benchmarks for " + f"problem {problem_names[0]}"
        if len(problem_names) == 1
        else f"problems {problem_names[0]} -> {problem_names[-1]}"
    )
    pbar = tqdm(desc=desc, total=n_pbar_units, leave=True)

    # --- split -------------------------------------------
    # parallel runs each use all cores themselves, so they run one at a time in this process
    # (spawning their own workers) instead of occupying a slot in the pool
    single_scope = [params for params in scope if not params.is_parallel]
    parallel_scope = [params for params in scope if params.is_parallel]

    # --- execute single-worker runs ----------------------
    # spawn, never fork: the parent has usually run numba parallel code by now (the distance-store
    # builds), and numba's threading layer is not fork-safe — forked children deadlock on their
    # first parallel call.  spawn starts workers clean on every platform and Python version.
    results = []
    if single_scope:
        with multiprocessing.get_context("spawn").Pool(processes=n_processes) as pool:
            for result in pool.imap_unordered(_execute_single_run, single_scope):
                results.append(result)
                pbar.n += get_pbar_units(result.params)
                pbar.refresh()

    # --- execute parallel runs ---------------------------
    for params in parallel_scope:
        results.append(_execute_single_run(params))
        pbar.n += get_pbar_units(params)
        pbar.refresh()

    # --- wrap up -----------------------------------------
    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    return results


# =================================================================================================
#  Execute SINGLE run
# =================================================================================================
def _execute_single_run(params: SolverPresetBenchmarkParams) -> SolverPresetBenchmarkResult:
    # --- init --------------------------------------------
    t_start = datetime.datetime.now().timestamp()

    # --- construct problem (untimed) ---------------------
    problem = BenchmarkProblemFactory.construct_problem(
        name=params.problem_name,
        n=params.problem_size,
        diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION,
    )

    # --- build & solve (end-to-end timed) ----------------
    with measure_end_to_end() as timing:
        if params.is_parallel:
            solver = (
                ParallelMaxDivSolverBuilder(problem)
                .with_workers(target_duration=params.duration, workers=params.n_workers)
                .with_seed(params.seed)
                .build()
            )
        else:
            solver = (
                MaxDivSolverBuilder(problem)
                .with_preset(target_duration=params.duration, preset=params.preset)
                .with_seed(params.seed)
                .build()
            )
        result = solver.solve(verbosity=Verbosity.SILENT)

    # --- return result -----------------------------------
    return SolverPresetBenchmarkResult(
        params=params,
        execution_info=SolverPresetBenchmarkExecutionInfo(
            pid=os.getpid(),
            t_start=t_start,
            t_end=datetime.datetime.now().timestamp(),
        ),
        t_elapsed_sec=timing.t_elapsed_sec,
        n_iterations=result.duration.n_iterations,
        score=result.score,
    )
