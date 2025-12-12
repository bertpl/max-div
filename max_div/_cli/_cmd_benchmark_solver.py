import click

from max_div.benchmarks import BenchmarkProblemFactory

from ._cmd_benchmark import benchmark


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
@solver.group(name="run")
@click.argument("test_problem")
@click.option(
    "--markdown",
    is_flag=True,
    default=False,
    help="Output benchmark results in Markdown table format.",
)
def run(test_problem: str, markdown: bool):
    """Run specific solver benchmark problem."""
    pass
