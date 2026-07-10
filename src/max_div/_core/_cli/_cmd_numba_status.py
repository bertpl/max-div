import click

from ._cli import cli


@cli.command()
def numba_status() -> None:
    """Show Numba version, llvmlite version, and configuration including SVML status."""
    import llvmlite
    import numba

    click.echo(f"Numba version    : {numba.__version__}")
    click.echo(f"llvmlite version : {llvmlite.__version__}")

    # Show key configuration settings
    from numba import config

    click.echo("\nNumba Configuration:")
    click.echo("-" * 50)
    # NOTE: numba.core.config members are generated dynamically at import time, so they are
    #       invisible to static type checkers — hence the per-line suppressions below.
    click.echo(f"SVML enabled       : {config.USING_SVML}")  # ty: ignore[unresolved-attribute]
    click.echo(f"Threading layer    : {config.THREADING_LAYER}")  # ty: ignore[unresolved-attribute]
    click.echo(f"Number of threads  : {config.NUMBA_NUM_THREADS}")  # ty: ignore[unresolved-attribute]
    click.echo(f"Optimization level : {config.OPT}")  # ty: ignore[unresolved-attribute]
    click.echo(f"Debug mode         : {config.DEBUG}")  # ty: ignore[unresolved-attribute]
    click.echo(f"Disable JIT        : {config.DISABLE_JIT}")  # ty: ignore[unresolved-attribute]
    click.echo("-" * 50)
