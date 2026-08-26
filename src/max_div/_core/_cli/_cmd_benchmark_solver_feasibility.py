import click

from ._cmd_benchmark_solver import solver
from .benchmarks.solver_feasibility import run_solver_feasibility_benchmark


# =================================================================================================
#  benchmark solver - feasibility
# =================================================================================================
@solver.command(name="feasibility")
@click.option(
    "--problem",
    is_flag=False,
    default="all",
    help="Constrained problem to generate verdicts for.",
)
@click.option(
    "--file",
    is_flag=True,
    default=False,
    help="Redirect output from console to one markdown file per problem.",
)
@click.option(
    "--markdown",
    is_flag=True,
    default=False,
    help="Output verdict tables in Markdown format.",
)
@click.option(
    "--turbo",
    is_flag=True,
    default=False,
    help="Run a fast smoke version with fewer sizes and a reduced budget; intended for testing purposes.",
)
def feasibility(problem: str, file: bool, markdown: bool, turbo: bool) -> None:
    """Generate certified per-size feasibility verdicts for the constrained benchmark problems."""
    run_solver_feasibility_benchmark(name=problem, markdown=markdown, file=file, turbo=turbo)
