import click

from max_div.internal.benchmarking._platform_speed import estimate_platform_speed

from ._cmd_benchmark import benchmark


# =================================================================================================
#  benchmark platform
# =================================================================================================
@benchmark.command(name="platform")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show details of 10 micro-benchmarks.",
)
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    help="Run quick, less reliable benchmark.",
)
def platform(verbose: bool, fast: bool):
    """Benchmark hardware / software combination (cpu, ram, os, python, numba, ...)."""
    click.echo("Benchmarking platform speed... ", nl=verbose)
    if verbose:
        click.echo("-" * 80)
        speed = estimate_platform_speed(silent=False, fast=fast)
        click.echo("-" * 80)
    else:
        speed = estimate_platform_speed(silent=True, fast=fast)
        click.echo("Done.")
    click.echo(f"Platform speed relative to reference platform: {speed:.3f}")
