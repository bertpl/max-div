import click

from ._cmd_benchmark import benchmark
from .bm_internal import (
    benchmark_diversity_metrics,
    benchmark_modify_p_selectivity,
    benchmark_randint,
    benchmark_randint_constrained,
)


# =================================================================================================
#  benchmark internal
# =================================================================================================
@benchmark.group(name="internal")
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
@click.pass_context
def internal(ctx, turbo: bool, speed: float, markdown: bool):
    """Internal benchmarks for max-div implementation details."""
    # Store flags in context so subcommands can access them
    ctx.ensure_object(dict)
    if turbo:
        ctx.obj["speed"] = 1.0
    else:
        ctx.obj["speed"] = speed
    ctx.obj["markdown"] = markdown


@internal.command(name="randint")
@click.pass_context
def randint(ctx):
    """Benchmarks the `randint` function from `max_div.sampling.uncon`."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    benchmark_randint(speed=speed, markdown=markdown)


@internal.command(name="randint_constrained")
@click.pass_context
def randint_constrained(ctx):
    """Benchmarks the `randint_constrained` function from `max_div.sampling.con`."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    benchmark_randint_constrained(speed=speed, markdown=markdown)


@internal.command(name="diversity_metrics")
@click.pass_context
def diversity_metrics(ctx):
    """Benchmarks computation of DiversityMetrics."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    benchmark_diversity_metrics(speed=speed, markdown=markdown)


@internal.command(name="modify_p_selectivity")
@click.pass_context
def modify_p_selectivity(ctx):
    """Benchmark different modify_p_selectivity flavors."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    benchmark_modify_p_selectivity(speed=speed, markdown=markdown)
