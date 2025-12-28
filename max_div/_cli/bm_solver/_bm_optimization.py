from max_div._cli.formatting import format_table_as_markdown, format_table_for_console
from max_div.internal.formatting import md_multiline
from max_div.solver import DiversityMetric, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._duration import iterations
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy
from max_div.solver._strategies._optimization._presets import OptimPreset

from ._base import BenchmarkSolverConstructor


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
            problem_name=problem_name,
            diversity_metric=diversity_metric,
        )
        self._n_iterations = n_iterations
        self._presets: dict[str, OptimPreset] = {
            str(preset.value): preset
            for preset in OptimPreset.all()
            if preset.is_relevant_for_problem(self.has_constraints)
        }

    def estimated_duration(self, size: int, strat_name: str) -> int:
        d, n, k, m, n_con_indices = self.get_problem_dimensions(size)
        preset = self._presets[strat_name]
        time_model = preset.time_model()
        estimated_duration_sec_per_iter = time_model.get_time_sec(n=n, k=k, m=m, n_con_indices=n_con_indices)
        estimated_duration_msec = max(1, round(1000 * self._n_iterations * estimated_duration_sec_per_iter))  # >=1 ms
        return estimated_duration_msec

    def construct_solver(self, size: int, strat_name: str, seed: int) -> MaxDivSolver:
        problem = self.construct_problem(size)
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

    def show_strategies_table(self, markdown: bool):
        # --- prepare table data ------------------------------
        if markdown:
            headers = ["`name`", "`class`", "`params`"]
        else:
            headers = ["name", "class", "params"]

        if self.has_constraints:
            headers.append("Constraint-aware")

        table_data = []
        for name, preset in self._presets.items():
            class_name = preset.class_name()
            if markdown:
                class_kwargs = md_multiline([f"{k}={str(v)}" for k, v in preset.class_kwargs().items()])
            else:
                class_kwargs = ", ".join([f"{k}={str(v)}" for k, v in preset.class_kwargs().items()])

            table_data.append(
                [
                    f"`{name}`" if markdown else name,
                    class_name,
                    class_kwargs,
                ]
            )
            if self.has_constraints:
                table_data[-1].append(preset.is_constraint_aware())

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
