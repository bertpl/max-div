from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import DiversityMetric

from ._base import SolverBenchmarkExecutor, SolverBenchmarkScope
from ._bm_initialization import BenchmarkSolverConstructor_Initialization
from ._bm_optimization import BenchmarkSolverConstructor_Optimization


def run_solver_benchmark(name: str, markdown: bool, file: bool = False, speed: float = 0.0):
    # -------------------------------------------------------------------------
    #  Special case
    # -------------------------------------------------------------------------
    if name == "all":
        # special case: run all benchmark problems
        all_problem_names = list(BenchmarkProblemFactory.get_all_benchmark_problems().keys())
        for problem_name in all_problem_names:
            run_solver_benchmark(problem_name, markdown, file, speed)
        return

    # -------------------------------------------------------------------------
    #  Regular case
    # -------------------------------------------------------------------------

    # --- initialization ----------------------------------
    executor = SolverBenchmarkExecutor(
        scope=SolverBenchmarkScope(
            solver_constructor=BenchmarkSolverConstructor_Initialization(
                problem_name=name, diversity_metric=DiversityMetric.geomean_separation()
            ),
            speed=speed,
            leave_pbar=file,
        )
    )
    executor.execute(markdown, file)

    # --- optimization ------------------------------------
    executor = SolverBenchmarkExecutor(
        scope=SolverBenchmarkScope(
            solver_constructor=BenchmarkSolverConstructor_Optimization(
                problem_name=name, diversity_metric=DiversityMetric.geomean_separation()
            ),
            speed=speed,
            leave_pbar=file,
        )
    )
    executor.execute(markdown, file)
