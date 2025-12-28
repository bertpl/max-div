import click

from max_div.benchmarks import BenchmarkProblemFactory

from ._cmd_benchmark import benchmark
from .bm_solver import run_solver_benchmark


# =================================================================================================
#  benchmark solver
# =================================================================================================
@benchmark.group(name="solver")
def solver():
    """Solver benchmarking functionality, based on built-in benchmark problems."""
    pass


# =================================================================================================
#  benchmark solver list
# =================================================================================================
@solver.command(name="list")
def _list():
    """List available test problems."""
    problem_classes = BenchmarkProblemFactory.get_all_benchmark_problems()
    click.echo("Available benchmark problems:")
    for name, cls in problem_classes.items():
        click.echo(f"- {name}: {cls.description()}")


# =================================================================================================
#  benchmark solver run
# =================================================================================================
@solver.command(name="run")
@click.argument("test_problem")
@click.option(
    "--file",
    is_flag=True,
    default=False,
    help="Redirect output from console to file.",
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
    "--markdown",
    is_flag=True,
    default=False,
    help="Output benchmark results in Markdown table format.",
)
@click.option(
    "--optimization-only",
    is_flag=True,
    default=False,
    help="Run only optimization benchmarks.",
)
@click.option(
    "--initialization-only",
    is_flag=True,
    default=False,
    help="Run only initialization benchmarks.",
)
def run(
    test_problem: str,
    file: bool,
    turbo: bool,
    speed: float,
    markdown: bool,
    optimization_only: bool,
    initialization_only: bool,
):
    """Run specific solver benchmark problem."""

    # --- argument handling -------------------------------
    if turbo:
        speed = 1.0
    if optimization_only:
        benchmark_initialization = False
        benchmark_optimization = True
    elif initialization_only:
        benchmark_initialization = True
        benchmark_optimization = False
    else:
        benchmark_initialization = True
        benchmark_optimization = True

    # --- run benchmarks ----------------------------------
    run_solver_benchmark(
        name=test_problem,
        markdown=markdown,
        file=file,
        speed=speed,
        benchmark_initialization=benchmark_initialization,
        benchmark_optimization=benchmark_optimization,
    )
