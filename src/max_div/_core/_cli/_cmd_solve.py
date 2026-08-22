import click

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset, TargetDuration, Verbosity

from ._cli import cli


@cli.command(name="solve")
@click.argument("test_problem")
@click.option(
    "--iterations",
    help="Number of iterations. Use this or --seconds to indicate duration.  Default=100 iter.",
)
@click.option(
    "--seconds",
    help="Number of seconds. Use this or --iterations to indicate duration.  Default=100 iter.",
)
@click.option(
    "--total-seconds",
    help="Wall-clock budget in seconds for the whole solve, distance computation and initialization included.",
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
    """Run the solver on requested benchmark problem."""
    # --- argument handling ----------------------
    if sum(option is not None for option in (iterations, seconds, total_seconds)) > 1:
        raise click.UsageError("Please provide only one of --iterations, --seconds or --total-seconds.")
    end_to_end_budget = total_seconds is not None
    if total_seconds is not None:
        duration = TargetDuration.seconds(float(total_seconds))
    elif iterations is not None:
        duration = TargetDuration.iterations(int(iterations))
    elif seconds is not None:
        duration = TargetDuration.seconds(float(seconds))
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
        .with_preset(target_duration=duration, preset=SolverPreset(preset), end_to_end_budget=end_to_end_budget)
        .build()
    )

    # --- solve ----------------------------------
    solver.solve(verbosity=verbosity)
