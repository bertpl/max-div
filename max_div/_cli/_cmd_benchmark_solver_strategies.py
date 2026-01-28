import click

from ._cmd_benchmark_solver import solver
from .bm_solver_strategies import run_solver_strategies_benchmark


# =================================================================================================
#  benchmark solver - strategies
# =================================================================================================
@solver.command(name="strategies")
@click.option(
    "--problem",
    is_flag=False,
    default="all",
    help="Problem to benchmark strategies on",
)
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
def strategies(
    problem: str,
    file: bool,
    turbo: bool,
    speed: float,
    markdown: bool,
    optimization_only: bool,
    initialization_only: bool,
):
    """Benchmark initialization/optimization strategies on specific solver benchmark problem."""

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
    run_solver_strategies_benchmark(
        name=problem,
        markdown=markdown,
        file=file,
        speed=speed,
        benchmark_initialization=benchmark_initialization,
        benchmark_optimization=benchmark_optimization,
    )
