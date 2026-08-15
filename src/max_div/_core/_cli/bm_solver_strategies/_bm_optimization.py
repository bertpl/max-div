from max_div._core._markdown import ReportElement
from max_div._core.metrics import DiversityMetric
from max_div._core.solver import MaxDivSolver, MaxDivSolverBuilder
from max_div._core.solver._duration import iterations
from max_div._core.solver._solver_step import OptimizationStep
from max_div._core.solver._strategies import InitializationStrategy

from ._base import BenchmarkSolverConstructor
from .presets import OptimPreset


# =================================================================================================
#  Main class
# =================================================================================================
class BenchmarkSolverConstructor_Optimization(BenchmarkSolverConstructor):
    def __init__(
        self,
        problem_name: str,
        diversity_metric: DiversityMetric = DiversityMetric.GEOMEAN_SEPARATION,
        n_iterations: int = 1000,
    ) -> None:
        super().__init__(
            benchmark_type="optimization",
            problem_name=problem_name,
            diversity_metric=diversity_metric,
        )
        self._n_iterations = n_iterations
        self._presets: dict[str, OptimPreset] = {
            str(preset.value): preset
            for preset in OptimPreset.all()
            if preset.is_relevant_for_problem(self.has_constraints)
        }

    def construct_solver(self, n: int, strat_name: str, seed: int) -> MaxDivSolver:
        problem = self.construct_problem(n)
        preset = self._presets[strat_name]
        return (
            MaxDivSolverBuilder(problem)
            .set_initialization_strategy(InitializationStrategy.fast())
            .add_solver_step(
                OptimizationStep(
                    optim_strategy=preset.create(),
                    duration=iterations(self._n_iterations),
                )
            )
            .with_seed(seed)
            .build()
        )

    def strategy_names(self) -> list[str]:
        return list(self._presets.keys())

    def build_strategies_table(self) -> list[ReportElement | str]:
        return self._strategies_table(
            f"Tested Optimization strategies ({self._n_iterations} iterations):", dict(self._presets)
        )
