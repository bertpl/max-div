import click

from ._cli import cli


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
