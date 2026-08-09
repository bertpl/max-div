"""A builder configures a solve: one builds a single solver, the other a portfolio of them."""

from ._base import SolverBuilderBase
from ._parallel import ParallelMaxDivSolverBuilder
from ._solver import MaxDivSolverBuilder

__all__ = [
    "MaxDivSolverBuilder",
    "ParallelMaxDivSolverBuilder",
    "SolverBuilderBase",
]
