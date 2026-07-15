"""Benchmark problem construction on top of max-div's built-in problem generators."""

from max_div.benchmark_problems import BenchmarkProblemFactory
from max_div.metrics import DiversityMetric
from max_div.problem import VectorMaxDivProblem


def build_problem(name: str, size: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
    """Construct a built-in benchmark problem (U1-U4 / C1-C4) at the given size.

    Args:
        name: Registered benchmark problem name, e.g. ``"U1"`` or ``"C3"``.
        size: The generator's size parameter ``s`` (n scales as ~100s).
        diversity_metric: Diversity metric the constructed problem optimizes.

    Returns:
        The constructed problem instance.
    """
    return BenchmarkProblemFactory.construct_problem(name, size=size, diversity_metric=diversity_metric)
