import click

from ._cli import cli
from .bm_internal import (
    benchmark_diversity_metrics,
    benchmark_modify_p_selectivity,
    benchmark_randint,
    benchmark_randint_constrained,
)


# =================================================================================================
#  benchmark
# =================================================================================================
@cli.group()
def benchmark():
    """Benchmarking commands."""
    pass
