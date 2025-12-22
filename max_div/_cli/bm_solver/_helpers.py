from functools import partial
from typing import Callable

from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import DiversityMetric, MaxDivProblem
from max_div.solver._strategies import InitializationStrategy


# =================================================================================================
#  Initialization strategies
# =================================================================================================
def get_initialization_strategies(constraints: bool) -> list[tuple[str, str, Callable[[], InitializationStrategy]]]:
    """
    Construct a list of initialization strategies based on whether the problem has constraints.
    Result is returns as a list of (name, description, strategy_factory_method) tuples.
    """
    result = []

    # --- InitRandomOneShot -------------------------------
    result.extend(
        [
            (
                "ROS(u)",
                "InitRandomOneShot(uniform=True, constrained=False)",
                partial(InitializationStrategy.random_one_shot, uniform=True, constrained=False),
            ),
            (
                "ROS(nu)",
                "InitRandomOneShot(uniform=False, constrained=False)",
                partial(InitializationStrategy.random_one_shot, uniform=False, constrained=False),
            ),
        ]
    )
    if constraints:
        result.extend(
            [
                (
                    "ROS(u,con)",
                    "InitRandomOneShot(uniform=True, constrained=True)",
                    partial(InitializationStrategy.random_one_shot, uniform=True, constrained=True),
                ),
                (
                    "ROS(nu,con)",
                    "InitRandomOneShot(uniform=False, constrained=True)",
                    partial(InitializationStrategy.random_one_shot, uniform=False, constrained=True),
                ),
            ]
        )

    # --- return ------------------------------------------
    return result


# =================================================================================================
#  Optimization strategies
# =================================================================================================


# =================================================================================================
#  Problem construction & properties
# =================================================================================================
def problem_has_constraints(name: str, size_range: list[int]) -> bool:
    """Determine if a benchmark problem has constraints based on its name and size range."""
    m_values = [
        construct_problem_instance(
            name=name,
            size=size,
            diversity_metric=DiversityMetric.geomean_separation(),
        ).m
        for size in size_range
    ]
    return max(m_values) > 0


def construct_problem_instance(name: str, size: int, diversity_metric: DiversityMetric) -> MaxDivProblem:
    """
    Construct a benchmark problem instance.
    In case some problem types have different parameters, we can encapsulate that logic here.
    """
    return BenchmarkProblemFactory.construct_problem(
        name=name,
        size=size,
        diversity_metric=diversity_metric,
    )
