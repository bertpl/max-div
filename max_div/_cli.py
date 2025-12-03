"""Command-line interface for max-div."""

import click

from max_div.benchmark import benchmark_diversity_metrics as _benchmark_diversity_metrics
from max_div.benchmark import benchmark_randint as _benchmark_randint
from max_div.benchmark import benchmark_randint_constrained as _benchmark_randint_constrained


# -------------------------------------------------------------------------
#  Main CLI Group
# -------------------------------------------------------------------------
@click.group()
def cli():
    """max-div: Flexible Solver for Maximum Diversity Problems with Fairness Constraints."""
    pass


# -------------------------------------------------------------------------
#  Benchmarking Commands
# -------------------------------------------------------------------------
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


@benchmark.command(name="randint")
@click.pass_context
def randint(ctx):
    """Benchmarks the `randint` function from `max_div.sampling.uncon`."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    _benchmark_randint(speed=speed, markdown=markdown)


@benchmark.command(name="randint_constrained")
@click.pass_context
def randint_constrained(ctx):
    """Benchmarks the `randint_constrained` function from `max_div.sampling.con`."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    _benchmark_randint_constrained(speed=speed, markdown=markdown)


@benchmark.command(name="diversity_metrics")
@click.pass_context
def diversity_metrics(ctx):
    """Benchmarks computation of DiversityMetrics."""
    speed = ctx.obj["speed"]
    markdown = ctx.obj["markdown"]
    _benchmark_diversity_metrics(speed=speed, markdown=markdown)


# -------------------------------------------------------------------------
#  Misc Commands
# -------------------------------------------------------------------------
@cli.command()
def numba_status():
    """Show Numba version, llvmlite version, and configuration including SVML status."""
    import llvmlite
    import numba

    click.echo(f"Numba version    : {numba.__version__}")
    click.echo(f"llvmlite version : {llvmlite.__version__}")

    # Show key configuration settings
    from numba import config

    click.echo("\nNumba Configuration:")
    click.echo("-" * 50)
    click.echo(f"SVML enabled       : {config.USING_SVML}")
    click.echo(f"Threading layer    : {config.THREADING_LAYER}")
    click.echo(f"Number of threads  : {config.NUMBA_NUM_THREADS}")
    click.echo(f"Optimization level : {config.OPT}")
    click.echo(f"Debug mode         : {config.DEBUG}")
    click.echo(f"Disable JIT        : {config.DISABLE_JIT}")
    click.echo("-" * 50)


# -------------------------------------------------------------------------
#  Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cli()
