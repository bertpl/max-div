from max_div.benchmarks import BenchmarkProblemFactory

from .benchmark_initialization import benchmark_initialization_strategies
from .benchmark_optimization import benchmark_optimization_strategies


def run_solver_benchmark(name: str, markdown: bool, file: bool = False, speed: float = 0.0):
    if name == "all":
        # special case: run all benchmark problems
        all_problem_names = list(BenchmarkProblemFactory.get_all_benchmark_problems().keys())
        for problem_name in all_problem_names:
            run_solver_benchmark(problem_name, markdown, file, speed)
    else:
        benchmark_initialization_strategies(name, markdown, file, speed)
        benchmark_optimization_strategies(name, markdown, file, speed)
