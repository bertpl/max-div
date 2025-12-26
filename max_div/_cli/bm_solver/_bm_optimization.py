from dataclasses import dataclass
from typing import Callable

from max_div._cli.formatting import format_table_as_markdown, format_table_for_console
from max_div.solver import DiversityMetric, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._duration import iterations
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy

from ._base import BenchmarkProblemGenerator, BenchmarkSolverConstructor


# =================================================================================================
#  Main class
# =================================================================================================
class BenchmarkSolverConstructor_Optimization(BenchmarkSolverConstructor):
    def __init__(
        self,
        problem_name: str,
        diversity_metric: DiversityMetric = DiversityMetric.geomean_separation(),
        n_iterations: int = 1000,
    ):
        super().__init__(
            benchmark_type="optimization",
            problem_generator=BenchmarkProblemGenerator(problem_name, diversity_metric),
        )
        self._n_iterations = n_iterations
        self._strategies: dict[str, OptimStrategyInfo] = {
            info.name: info for info in get_optimization_strategies(self.has_constraints)
        }

    def estimated_duration(self, size: int, strat_name: str) -> int:
        return size  # optimization methods (if # iterations are fixed) scale with O(k+n) = O(size)

    def construct_solver(self, size: int, strat_name: str, seed: int) -> MaxDivSolver:
        problem = self.construct_problem(size)
        strat_info = self._strategies[strat_name]
        return (
            MaxDivSolverBuilder(problem)
            .set_initialization_strategy(InitializationStrategy.dummy())
            .add_solver_step(
                OptimizationStep(
                    optim_strategy=strat_info.factory(),
                    duration=iterations(self._n_iterations),
                )
            )
            .with_seed(seed)
            .build()
        )

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

        print(f"Tested Optimization strategies ({self._n_iterations} iterations):")
        print()
        for line in display_data:
            print(line)
        print()


# =================================================================================================
#  Helper classes
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
