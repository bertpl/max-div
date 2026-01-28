import click

from max_div.benchmarks import BenchmarkProblemFactory

from ._cmd_benchmark_solver import solver


# =================================================================================================
#  benchmark solver - list problems
# =================================================================================================
@solver.command(name="list_problems")
def list_problems():
    """List available test problems."""
    problem_classes = BenchmarkProblemFactory.get_all_benchmark_problems()
    click.echo("Available benchmark problems:")
    for name, cls in problem_classes.items():
        click.echo(f"- {name}: {cls.description()}")
