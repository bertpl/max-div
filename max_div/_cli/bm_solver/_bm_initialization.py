from dataclasses import dataclass
from functools import partial
from typing import Callable

from max_div._cli.formatting import format_table_as_markdown, format_table_for_console
from max_div.solver import DiversityMetric, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._strategies import InitializationStrategy

from ._base import BenchmarkProblemGenerator, BenchmarkSolverConstructor


# =================================================================================================
#  Main class
# =================================================================================================
class BenchmarkSolverConstructor_Initialization(BenchmarkSolverConstructor):
    def __init__(self, problem_name: str, diversity_metric: DiversityMetric = DiversityMetric.geomean_separation()):
        super().__init__(
            benchmark_type="initialization",
            problem_generator=BenchmarkProblemGenerator(problem_name, diversity_metric),
        )
        self._strategies: dict[str, InitStrategyInfo] = {
            info.name: info for info in get_initialization_strategies(self.has_constraints)
        }

    def estimated_duration(self, size: int, strat_name: str) -> int:
        return size * size  # initialization methods roughly scale with O(kn) = O(size^2)

    def construct_solver(self, size: int, strat_name: str, seed: int) -> MaxDivSolver:
        problem = self.construct_problem(size)
        strat_info = self._strategies[strat_name]
        return MaxDivSolverBuilder(problem).set_initialization_strategy(strat_info.factory()).with_seed(seed).build()

    def strategy_names(self) -> list[str]:
        return list(self._strategies.keys())

    def show_strategies_table(self, markdown: bool):
        # --- prepare table data ------------------------------
        if markdown:
            headers = ["`name`", "`class`", "`params`"]
        else:
            headers = ["name", "class", "params"]

        if self.has_constraints:
            headers.append("Constraint-aware")

        table_data = []
        for strat_info in self._strategies.values():
            table_data.append(
                [
                    f"`{strat_info.name}`" if markdown else strat_info.name,
                    strat_info.class_name,
                    strat_info.class_kwargs,
                ]
            )
            if self.has_constraints:
                table_data[-1].append(strat_info.uses_constraints)

        # --- show table ---
        if markdown:
            display_data = format_table_as_markdown(headers, table_data)
        else:
            display_data = format_table_for_console(headers, table_data)

        print("Tested Initialization strategies:")
        print()
        for line in display_data:
            print(line)
        print()


# =================================================================================================
#  Helper classes
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
