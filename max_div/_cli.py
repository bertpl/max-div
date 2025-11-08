"""Command-line interface for max-div."""

import click

from max_div.benchmark import benchmark_sample_int as _benchmark_sample_int


@click.group()
def cli():
    """max-div: Flexible Solver for Maximum Diversity Problems with Fairness Constraints."""
    pass


@cli.group()
@click.option(
    "--turbo",
    is_flag=True,
    default=False,
    help="Run a much shorter (but less reliable) benchmark; intended for testing purposes.",
)
@click.option(
    "--markdown",
    is_flag=True,
    default=False,
    help="Output benchmark results in Markdown table format.",
)
@click.pass_context
def benchmark(ctx, turbo: bool, markdown: bool):
    """Benchmarking commands."""
    # Store flags in context so subcommands can access them
    ctx.ensure_object(dict)
    ctx.obj["turbo"] = turbo
    ctx.obj["markdown"] = markdown


@benchmark.command()
@click.pass_context
def sample_int(ctx):
    """Benchmarks the `sample_int` function from `max_div.sampling.discrete`."""
    turbo = ctx.obj["turbo"]
    markdown = ctx.obj["markdown"]
    _benchmark_sample_int(turbo=turbo, markdown=markdown)


if __name__ == "__main__":
    cli()
