from dataclasses import dataclass, field
from pathlib import Path

from max_div.benchmarks import BenchmarkProblemFactory
from max_div.internal.markdown import Report, Table, TableElement, TableTimeElapsed, TableValueWithUncertainty, h1, h2
from max_div.internal.utils import stdout_to_file

from ._models import SolverPresetBenchmarkResult


# =================================================================================================
#  Helpers
# =================================================================================================
@dataclass
class AggregateResult:
    # aggregation of results for a single (problem, preset, target_duration)-combination
    elapsed_sec: list[float] = field(default_factory=list)
    n_iterations: list[int] = field(default_factory=list)
    constraint_scores: list[float] = field(default_factory=list)
    diversity_scores: list[float] = field(default_factory=list)


# =================================================================================================
#  Main function
# =================================================================================================
def show_solver_presets_benchmark_results(
    results: list[SolverPresetBenchmarkResult],
    markdown: bool,
    markdown_file_name: str | None,
):
    # --- extract scope -----------------------------------
    problem_size = results[0].params.problem_size
    problem_names = sorted({result.params.problem_name for result in results})
    presets = sorted({result.params.preset for result in results})
    target_durations = sorted({result.params.duration for result in results})

    # --- aggregate & show --------------------------------
    headers = ["Target duration"] + [p.value.upper() for p in presets]
    for problem_name in problem_names:
        # check problem properties
        _, _, _, m, _ = BenchmarkProblemFactory.get_problem_dimensions(problem_name, size=problem_size)
        has_constraints = m > 0

        # each problem has its own tables with results
        table_actual_times = Table(headers)  # table with actual elapsed times for each (target_duration, preset)
        table_con_score = Table(headers)  # table with constraint scores for each (target_duration, preset)
        table_div_score = Table(headers)  # table with diversity scores for each (target_duration, preset)
        table_iters = Table(headers)  # table with iteration counts for each (target_duration, preset)

        for target_duration in target_durations:
            # each duration has its own row
            table_actual_times_row: list[TableElement | str] = [str(target_duration)]
            table_con_score_row: list[TableElement | str] = [str(target_duration)]
            table_div_score_row: list[TableElement | str] = [str(target_duration)]
            table_iters_row: list[TableElement | str] = [str(target_duration)]

            for preset in presets:
                # each preset has its own column

                # build aggregate result
                agg = AggregateResult()
                for result in results:
                    if (result.params.problem_name, result.params.duration, result.params.preset) == (
                        problem_name,
                        target_duration,
                        preset,
                    ):
                        agg.elapsed_sec.append(result.execution_info.elapsed_sec)
                        agg.n_iterations.append(result.n_iterations)
                        agg.constraint_scores.append(result.score.constraints)
                        agg.diversity_scores.append(result.score.diversity)

                # add results for this preset to table rows
                table_actual_times_row.append(TableTimeElapsed.from_values(agg.elapsed_sec))
                table_con_score_row.append(
                    TableValueWithUncertainty.from_values(
                        agg.constraint_scores,
                        decimals=6,
                    )
                )
                table_div_score_row.append(
                    TableValueWithUncertainty.from_values(
                        agg.diversity_scores,
                        decimals=6,
                        scientific=True,
                    )
                )
                table_iters_row.append(
                    TableValueWithUncertainty.from_values(
                        agg.n_iterations,
                        decimals=0,
                    )
                )

            # add rows to tables
            table_actual_times.add_row(table_actual_times_row)
            table_con_score.add_row(table_con_score_row)
            table_div_score.add_row(table_div_score_row)
            table_iters.add_row(table_iters_row)

        # finalize tables
        table_con_score.highlight_results(TableValueWithUncertainty, clr_highest=Table.GREEN, clr_lowest=Table.RED)
        table_div_score.highlight_results(TableValueWithUncertainty, clr_highest=Table.GREEN, clr_lowest=Table.RED)
        table_iters.highlight_results(TableValueWithUncertainty, clr_highest=Table.GREEN, clr_lowest=Table.RED)

        # build report for this problem
        report = Report()
        report += [
            h1(f"Benchmark Results for Problem '{problem_name}' (size={problem_size})."),
            h2("Actual Total Elapsed Times"),
            "(including initialization & fixed overhead)",
            table_actual_times,
        ]
        if has_constraints:
            report += [
                h2("Constraint Scores"),
                table_con_score,
            ]
        report += [
            h2("Diversity Scores"),
            table_div_score,
            h2("Number of Iterations"),
            table_iters,
        ]

        # output report
        with stdout_to_file(enabled=markdown_file_name is not None, filename=Path(str(markdown_file_name))):
            report.print(markdown=markdown)
