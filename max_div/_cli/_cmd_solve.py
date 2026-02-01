import click

from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import DiversityMetric, MaxDivSolverBuilder, SolverPreset, TargetDuration

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
    "--verbosity",
    default=20,
    help="Verbosity level (0=silent, 10=tqdm, 20=tabular). Default=20.",
)
@click.option(
    "--size",
    default=10,
    help="Problem size parameter. Default=10.",
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
    verbosity: int = 20,
    size: int = 10,
    preset: str = "default",
) -> None:
    """Run the solver on requested benchmark problem."""

    # --- argument handling -------------------------------
    if (iterations is not None) and (seconds is not None):
        raise click.UsageError("Please provide only one of --iterations or --seconds.")
    if (not iterations) and (not seconds):
        duration = TargetDuration.iterations(100)  # default to 100 iterations
    elif iterations is not None:
        duration = TargetDuration.iterations(int(iterations))
    else:
        duration = TargetDuration.seconds(float(seconds))

    # --- show what we'll do ------------------------------
    click.echo(
        f"Solving test problem '{test_problem}' for a duration of {str(duration)} using {preset.upper()} preset..."
    )

    # --- construct solver --------------------------------
    solver = (
        MaxDivSolverBuilder(
            BenchmarkProblemFactory.construct_problem(
                name=test_problem,
                size=size,
                diversity_metric=DiversityMetric.approx_geomean_separation(),
            ),
        )
        .with_preset(target_duration=duration, preset=SolverPreset(preset))
        .build()
    )

    # --- solve -------------------------------------------
    solver.solve(verbosity=verbosity)
