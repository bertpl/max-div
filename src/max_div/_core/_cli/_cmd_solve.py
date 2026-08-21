import click

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset, TargetDuration, Verbosity

from ._cli import cli


@cli.command(name="solve")
@click.argument("test_problem")
@click.option(
    "--iterations",
    help="Number of optimization iterations to run.",
)
@click.option(
    "--seconds",
    help="Number of seconds to optimize for, on top of building the distances and initializing.",
)
@click.option(
    "--total-seconds",
    help="Number of seconds the whole solve may take, distances and initialization included.",
)
@click.option(
    "--verbosity",
    default=20,
    help="Verbosity level (0=silent, 10=tqdm, 20=tabular). Default=20.",
)
@click.option(
    "--n",
    default=1000,
    help="Problem size n. Default=1000.",
)
@click.option(
    "--preset",
    default="default",
    help="Set solver preset to use. Default='default'. Options: " + ", ".join([p.value for p in SolverPreset]),
)
def solve(
    test_problem: str,
    iterations: int | None = None,
    seconds: float | None = None,
    total_seconds: float | None = None,
    verbosity: int = Verbosity.TABULAR,
    n: int = 1000,
    preset: str = "default",
) -> None:
    """Run the solver on requested benchmark problem, for 100 iterations unless a budget is given."""
    # --- argument handling ----------------------
    given_duration_flags = {
        flag: value
        for flag, value in (
            ("--iterations", iterations),
            ("--seconds", seconds),
            ("--total-seconds", total_seconds),
        )
        if value is not None
    }
    if len(given_duration_flags) > 1:
        raise click.UsageError(f"Please provide only one of {', '.join(given_duration_flags)}.")
    if iterations is not None:
        duration = TargetDuration.iterations(int(iterations))
    elif seconds is not None:
        duration = TargetDuration.seconds(float(seconds))
    elif total_seconds is not None:
        duration = TargetDuration.total_seconds(float(total_seconds))
    else:
        duration = TargetDuration.iterations(100)

    # --- show what we'll do ---------------------
    click.echo(f"Solving test problem '{test_problem}' for a duration of {duration!s} using {preset.upper()} preset...")

    # --- construct solver -----------------------
    solver = (
        MaxDivSolverBuilder(
            BenchmarkProblemFactory.construct_problem(
                name=test_problem,
                n=n,
                diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION,
            ),
        )
        .with_preset(target_duration=duration, preset=SolverPreset(preset))
        .build()
    )

    # --- solve ----------------------------------
    solver.solve(verbosity=verbosity)
