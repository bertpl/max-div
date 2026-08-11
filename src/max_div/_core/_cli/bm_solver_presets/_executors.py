import datetime
import multiprocessing
import os

from tqdm import tqdm

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.solver import MaxDivSolverBuilder, Verbosity

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

    # --- execute -----------------------------------------
    # spawn, never fork: the parent has usually run numba parallel code by now (the distance-store
    # builds), and numba's threading layer is not fork-safe — forked children deadlock on their
    # first parallel call.  spawn starts workers clean on every platform and Python version.
    results = []
    with multiprocessing.get_context("spawn").Pool(processes=n_processes) as pool:
        for result in pool.imap_unordered(_execute_single_run, scope):
            results.append(result)
            pbar.n += get_pbar_units(result.params)
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

    # --- construct solver --------------------------------
    solver = (
        MaxDivSolverBuilder(
            BenchmarkProblemFactory.construct_problem(
                name=params.problem_name,
                size=params.problem_size,
                diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION,
            ),
        )
        .with_preset(target_duration=params.duration, preset=params.preset)
        .with_seed(params.seed)
        .build()
    )

    # --- solve -------------------------------------------
    result = solver.solve(verbosity=Verbosity.SILENT)

    # --- return result -----------------------------------
    return SolverPresetBenchmarkResult(
        params=params,
        execution_info=SolverPresetBenchmarkExecutionInfo(
            pid=os.getpid(),
            t_start=t_start,
            t_end=datetime.datetime.now().timestamp(),
        ),
        t_elapsed_sec=result.duration.t_elapsed_sec,
        n_iterations=result.duration.n_iterations,
        score=result.score,
    )
