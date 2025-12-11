import click

from ._cli import cli
from .benchmarking import (
    benchmark_diversity_metrics,
    benchmark_modify_p_selectivity,
    benchmark_randint,
    benchmark_randint_constrained,
)


# =================================================================================================
#  Main benchmark command
# =================================================================================================
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


# =================================================================================================
#  Sub-commands
# =================================================================================================
@benchmark.command(name="randint")
@click.pass_context
def randint(ctx):
    """Benchmarks the `randint` function from `max_div.sampling.uncon`."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    benchmark_randint(speed=speed, markdown=markdown)


@benchmark.command(name="randint_constrained")
@click.pass_context
def randint_constrained(ctx):
    """Benchmarks the `randint_constrained` function from `max_div.sampling.con`."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    benchmark_randint_constrained(speed=speed, markdown=markdown)


@benchmark.command(name="diversity_metrics")
@click.pass_context
def diversity_metrics(ctx):
    """Benchmarks computation of DiversityMetrics."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    benchmark_diversity_metrics(speed=speed, markdown=markdown)


@benchmark.command(name="modify_p_selectivity")
@click.pass_context
def modify_p_selectivity(ctx):
    """Benchmark different modify_p_selectivity flavors."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    benchmark_modify_p_selectivity(speed=speed, markdown=markdown)
