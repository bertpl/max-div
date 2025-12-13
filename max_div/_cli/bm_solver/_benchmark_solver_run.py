from .benchmark_initialization import benchmark_initialization_strategies


def run_solver_benchmark(name: str, markdown: bool, speed: float = 0.0):
    benchmark_initialization_strategies(name, markdown, speed)
