import datetime
import os
from multiprocessing import Pool

from tqdm import tqdm

from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import DiversityMetric, MaxDivSolverBuilder

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
    pbar = tqdm(desc="Executing preset benchmarks", total=n_pbar_units, leave=True)

    # --- execute -----------------------------------------
    results = []
    with Pool(processes=n_processes) as pool:
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
                diversity_metric=DiversityMetric.approx_geomean_separation(),
            ),
        )
        .with_preset(target_duration=params.duration, preset=params.preset)
        .with_seed(params.seed)
        .build()
    )

    # --- solve -------------------------------------------
    result = solver.solve(verbosity=0)

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
