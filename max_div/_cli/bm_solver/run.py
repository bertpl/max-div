from .run_initialization import benchmark_initialization_strategies


def run_solver_benchmark(name: str, markdown: bool):
    benchmark_initialization_strategies(name, markdown)
