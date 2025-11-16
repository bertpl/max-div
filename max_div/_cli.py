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
    help="Run shorter, less accurate benchmark; identical to --speed=1.0; intended for testing purposes.",
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
@click.pass_context
def benchmark(ctx, turbo: bool, speed: float, markdown: bool):
    """Benchmarking commands."""
    # Store flags in context so subcommands can access them
    ctx.ensure_object(dict)
    if turbo:
        ctx.obj["speed"] = 1.0
    else:
        ctx.obj["speed"] = speed
    ctx.obj["markdown"] = markdown


@benchmark.command()
@click.pass_context
def sample_int(ctx):
    """Benchmarks the `sample_int` function from `max_div.sampling.discrete`."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    _benchmark_sample_int(speed=speed, markdown=markdown)


if __name__ == "__main__":
    cli()
