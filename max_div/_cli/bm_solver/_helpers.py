from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Callable

from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import DiversityMetric, MaxDivProblem
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  Initialization strategies
# =================================================================================================
@dataclass
class InitStrategyInfo:
    name: str
    class_name: str
    class_kwargs: str
    factory: Callable[[], InitializationStrategy]
    needs_constraints: bool
    uses_constraints: bool


def get_initialization_strategies(constraints: bool) -> list[InitStrategyInfo]:
    """
    Construct a list of initialization strategies based on whether the problem has constraints.
    Result is returns as a list of InitStrategyInfo objects.
    """
    result: list[InitStrategyInfo] = []

    # --- InitDummy ---------------------------------------
    result.extend(
        [
            InitStrategyInfo(
                name="REF",
                class_name="InitDummy",
                class_kwargs="/",
                factory=InitializationStrategy.dummy,
                needs_constraints=False,
                uses_constraints=False,
            ),
        ]
    )

    # --- InitRandomOneShot -------------------------------
    result.extend(
        [
            InitStrategyInfo(
                name="ROS(u)",
                class_name="InitRandomOneShot",
                class_kwargs="uniform=True, constrained=False",
                factory=partial(InitializationStrategy.random_one_shot, uniform=True, constrained=False),
                needs_constraints=False,
                uses_constraints=False,
            ),
            InitStrategyInfo(
                name="ROS(nu)",
                class_name="InitRandomOneShot",
                class_kwargs="uniform=False, constrained=False",
                factory=partial(InitializationStrategy.random_one_shot, uniform=False, constrained=False),
                needs_constraints=False,
                uses_constraints=False,
            ),
            InitStrategyInfo(
                name="ROS(u,con)",
                class_name="InitRandomOneShot",
                class_kwargs="uniform=True, constrained=True",
                factory=partial(InitializationStrategy.random_one_shot, uniform=True, constrained=True),
                needs_constraints=True,
                uses_constraints=True,
            ),
            InitStrategyInfo(
                name="ROS(nu,con)",
                class_name="InitRandomOneShot",
                class_kwargs="uniform=False, constrained=True",
                factory=partial(InitializationStrategy.random_one_shot, uniform=False, constrained=True),
                needs_constraints=True,
                uses_constraints=True,
            ),
        ]
    )

    # --- InitRandomBatched -------------------------------
    result.extend(
        [
            InitStrategyInfo(
                name="RB(2)",
                class_name="InitRandomBatched",
                class_kwargs="b=2, constrained=False",
                factory=partial(InitializationStrategy.random_batched, b=2, constrained=False),
                needs_constraints=False,
                uses_constraints=False,
            ),
            InitStrategyInfo(
                name="RB(10)",
                class_name="InitRandomBatched",
                class_kwargs="b=10, constrained=False",
                factory=partial(InitializationStrategy.random_batched, b=10, constrained=False),
                needs_constraints=False,
                uses_constraints=False,
            ),
            InitStrategyInfo(
                name="RB(2,con)",
                class_name="InitRandomBatched",
                class_kwargs="b=2, constrained=True",
                factory=partial(InitializationStrategy.random_batched, b=2, constrained=True),
                needs_constraints=True,
                uses_constraints=True,
            ),
            InitStrategyInfo(
                name="RB(10,con)",
                class_name="InitRandomBatched",
                class_kwargs="b=10, constrained=True",
                factory=partial(InitializationStrategy.random_batched, b=10, constrained=True),
                needs_constraints=True,
                uses_constraints=True,
            ),
        ]
    )

    # --- InitEager ---------------------------------------
    result.extend(
        [
            InitStrategyInfo(
                name="E(2)",
                class_name="InitEager",
                class_kwargs="nc=2",
                factory=partial(InitializationStrategy.eager, nc=2),
                needs_constraints=False,
                uses_constraints=True,
            ),
            InitStrategyInfo(
                name="E(10)",
                class_name="InitEager",
                class_kwargs="nc=10",
                factory=partial(InitializationStrategy.eager, nc=10),
                needs_constraints=False,
                uses_constraints=True,
            ),
        ]
    )

    # --- return ------------------------------------------
    return [info for info in result if (not info.needs_constraints) or constraints]


# =================================================================================================
#  Optimization strategies
# =================================================================================================
@dataclass
class OptimStrategyInfo:
    name: str
    class_name: str
    class_kwargs: str
    factory: Callable[[], OptimizationStrategy]
    needs_constraints: bool
    uses_constraints: bool


def get_optimization_strategies(constraints: bool) -> list[OptimStrategyInfo]:
    """
    Construct a list of optimization strategies based on whether the problem has constraints.
    Result is returns as a list of OptimStrategyInfo objects.
    """
    result: list[OptimStrategyInfo] = []

    # --- OptimRandomSwaps --------------------------------
    result.extend(
        [
            OptimStrategyInfo(
                name="REF",
                class_name="OptimRandomSwaps",
                class_kwargs="/",
                factory=OptimizationStrategy.random_swaps,
                needs_constraints=False,
                uses_constraints=False,
            ),
        ]
    )

    # --- return ------------------------------------------
    return [info for info in result if (not info.needs_constraints) or constraints]


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


def get_size_range(speed: float) -> list[int]:
    """Get size range based on speed parameter."""
    max_size = round(64 ** (1 - speed))  # 64 for speed=0.0 to 1 for speed=1.0
    return sorted({value for c, k in product([1.0, 1.5], range(50)) if (value := round(c * (2.0**k))) <= max_size})
