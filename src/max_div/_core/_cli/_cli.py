"""
Command-line interface for max-div.

This file contains the main CLI group; specific commands are implemented in files _cmd_*.py
"""

import click


@click.group()
def cli():
    """max-div: Flexible Solver for Maximum Diversity Problems with Fairness Constraints."""
    ...
