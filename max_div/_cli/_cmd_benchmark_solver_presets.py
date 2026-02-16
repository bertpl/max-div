from datetime import datetime, timedelta
from pathlib import Path

import click

from max_div.benchmarks import BenchmarkProblemFactory
from max_div.internal.formatting import format_time_duration
from max_div.solver import SolverPreset

from ._cmd_benchmark_solver import solver
from .bm_solver_presets import (
    determine_benchmark_scope,
    determine_benchmark_scope_for_max_duration,
    estimate_execution_time_sec_multi,
    execute_solver_presets_benchmark,
    get_n_processes,
    show_solver_presets_benchmark_results,
)


# =================================================================================================
#  benchmark solver - presets
# =================================================================================================
@solver.command(name="presets")
@click.option(
    "--preset",
    is_flag=False,
    default="all",
    help="Solver preset to benchmark",
)
@click.option(
    "--problem",
    is_flag=False,
    default="all",
    help="Problem to benchmark solver presets on",
)
@click.option(
    "--size",
    is_flag=False,
    default=100,
    help="Problem size to benchmark solver presets on",
)
@click.option(
    "--json-file",
    is_flag=True,
    default=False,
    help="Save benchmark results to json file.",
)
@click.option(
    "--markdown-file",
    is_flag=True,
    default=False,
    help="Save benchmark results to .md file, instead of writing report to terminal.",
)
@click.option(
    "--turbo",
    is_flag=True,
    default=False,
    help="Run shorter, less accurate/complete benchmark; identical to --speed=1.0; intended for testing purposes.",
)
@click.option(
    "--speed",
    default=0.0,
    help="Values closer to 1.0 result in shorter, less accurate benchmark; Overridden by --turbo when provided.",
)
@click.option(
    "--target-max-minutes",
    type=float,
    required=False,
    default=None,
    help="When provided, overrides --speed or --turbo setting and chooses speed parameter based on max duration.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="When True, determines & reports scope, but does not perform any benchmark.",
)
@click.option(
    "--max-run-duration-minutes",
    type=float,
    required=False,
    default=None,
    help="When provided, overrides the maximum duration of a single benchmark run.",
)
@click.option(
    "--markdown",
    is_flag=True,
    default=False,
    help="Output benchmark results in Markdown table format.",
)
def presets(
    preset: str,
    problem: str,
    size: int,
    json_file: bool,
    markdown_file: bool,
    turbo: bool,
    speed: float,
    target_max_minutes: float | None,
    dry_run: bool,
    max_run_duration_minutes: float | None,
    markdown: bool,
):
    """Benchmark solver presets on specific solver benchmark problem."""

    # --- argument handling - speed -----------------------
    if turbo:
        speed = 1.0
    if markdown_file:
        markdown = True

    # --- argument handling - preset(s) & problem(s) ------
    presets = resolve_presets(preset)
    problems = resolve_problems(problem)

    # --- argument handling - max_run_duration -------------
    max_run_duration_sec = 60.0 * max_run_duration_minutes if max_run_duration_minutes else None

    # --- determine scope ---------------------------------
    if target_max_minutes:
        speed, scope = determine_benchmark_scope_for_max_duration(
            presets=presets,
            problems=problems,
            size=size,
            max_duration_sec=60.0 * target_max_minutes,
            max_run_duration_sec=max_run_duration_sec,
        )
    else:
        scope = determine_benchmark_scope(
            presets=presets,
            problems=problems,
            size=size,
            speed=speed,
            max_run_duration_sec=max_run_duration_sec,
        )

    # --- report scope & estimated duration ---------------

    # gather statistics
    n_processes = get_n_processes(len(scope))
    n_durations = len({s.duration for s in scope})
    durations_sec = sorted({s.duration for s in scope})
    if len(durations_sec) > 4:
        durations_str = ", ".join(
            [
                str(durations_sec[0]),
                str(durations_sec[1]),
                "...",
                str(durations_sec[-2]),
                str(durations_sec[-1]),
            ]
        )
    else:
        durations_str = ", ".join([str(d) for d in durations_sec])
    n_seeds = len({s.seed for s in scope})
    min_seed = min([s.seed for s in scope])
    max_seed = max([s.seed for s in scope])
    est_duration_str = format_time_duration(estimate_execution_time_sec_multi(scope), n_chars=8).strip()
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=estimate_execution_time_sec_multi(scope))
    start_time_str = start_time.strftime("%a %Y-%m-%d %H:%M:%S")
    end_time_str = end_time.strftime("%a %Y-%m-%d %H:%M:%S")

    # report statistics
    click.echo(f"Executing {len(scope)} solver preset benchmark runs using {n_processes} parallel processes...")
    click.echo(f"  - problem size  : {size:_}")
    click.echo(f"  - speed         : {speed:.6f}")
    click.echo(f"  - problems      : {len(problems):_}".ljust(40) + f"[{', '.join(problems)}]")
    click.echo(f"  - presets       : {len(presets):_}".ljust(40) + f"[{', '.join(presets)}]")
    click.echo(f"  - durations     : {n_durations:_}".ljust(40) + f"[{durations_str}]")
    click.echo(f"  - seeds         : {n_seeds:_}".ljust(40) + f"[{min_seed} -> {max_seed}]")
    click.echo(f"  - est. duration : {est_duration_str}".ljust(40) + f"[{start_time_str} -> {end_time_str}]")

    # --- run benchmarks ----------------------------------
    if dry_run:
        click.echo("")
        click.echo("---=== DRY_RUN ENABLED - SKIPPING BENCHMARK EXECUTION ===---")
        click.echo("")
    else:
        for problem in problems:
            # file names for this problem
            json_file_name = f"preset_results_{problem}_{size}.json" if json_file else None
            markdown_file_name = f"preset_results_{problem}_{size}.md" if markdown_file else None

            # execute for this problem
            results = execute_solver_presets_benchmark(
                scope=[p for p in scope if p.problem_name == problem],
                json_file_name=json_file_name,
            )

            # show for this problem
            show_solver_presets_benchmark_results(
                results=results,
                markdown=markdown,
                markdown_file_name=markdown_file_name,
            )


# =================================================================================================
#  Helpers
# =================================================================================================
def resolve_presets(preset: str) -> list[SolverPreset]:
    """Resolve preset string into list of SolverPreset enums."""
    if preset == "all":
        presets: list[SolverPreset] = list(SolverPreset)
    elif "," in preset:
        presets: list[SolverPreset] = [SolverPreset(p.strip()).resolve_alias() for p in preset.split(",")]
    else:
        presets: list[SolverPreset] = [SolverPreset(preset)]
    return sorted({p.resolve_alias() for p in presets})


def resolve_problems(problem: str) -> list[str]:
    """Resolve problem string into list of problem names."""
    if problem == "all":
        return BenchmarkProblemFactory.get_all_benchmark_names()
    else:
        if "," in problem:
            problems = [p.strip() for p in problem.split(",")]
        else:
            problems = [problem]
        all_supported_problem_names = BenchmarkProblemFactory.get_all_benchmark_names()
        for problem in problems:
            if problem not in all_supported_problem_names:
                raise ValueError(f"Unknown problem name '{problem}'. Available problems: {all_supported_problem_names}")
        return problems
