from max_div._core._markdown import ReportElement, Table
from max_div._core.metrics import DiversityMetric
from max_div._core.solver import MaxDivSolver, MaxDivSolverBuilder

from ._base import BenchmarkSolverConstructor
from .presets import InitPreset


# =================================================================================================
#  Main class
# =================================================================================================
class BenchmarkSolverConstructor_Initialization(BenchmarkSolverConstructor):
    def __init__(self, problem_name: str, diversity_metric: DiversityMetric = DiversityMetric.GEOMEAN_SEPARATION):
        super().__init__(
            benchmark_type="initialization",
            problem_name=problem_name,
            diversity_metric=diversity_metric,
        )
        self._presets: dict[str, InitPreset] = {
            str(preset.value): preset
            for preset in InitPreset.all()
            if preset.is_relevant_for_problem(self.has_constraints)
        }

    def construct_solver(self, size: int, strat_name: str, seed: int) -> MaxDivSolver:
        problem = self.construct_problem(size)
        preset = self._presets[strat_name]
        return MaxDivSolverBuilder(problem).set_initialization_strategy(preset.create()).with_seed(seed).build()

    def strategy_names(self) -> list[str]:
        return list(self._presets.keys())

    def build_strategies_table(self) -> list[ReportElement | str]:
        # --- prepare table ---------------------
        table = Table(["`name`", "`class`", "`params`"] + (["Constraint-aware"] if self.has_constraints else []))

        for name, preset in self._presets.items():
            table.add_row(
                [
                    f"`{name}`",
                    preset.class_name(),
                    "\n".join([f"{k}={str(v)}" for k, v in preset.class_kwargs().items()]),
                ]
                + ([str(preset.is_constraint_aware())] if self.has_constraints else [])
            )

        # --- return ReportElements list --------
        return ["Tested Initialization strategies:", table]
