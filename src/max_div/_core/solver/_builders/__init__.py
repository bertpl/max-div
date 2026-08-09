"""Builders that configure a solve, one for a single solver and one for a portfolio of them.

`_base` holds what both carry — the problem, and the settings that define the score — while each
concrete builder adds the search it configures and the thing it builds.
"""

from ._base import SolverBuilderBase
from ._parallel import ParallelMaxDivSolverBuilder
from ._solver import MaxDivSolverBuilder

__all__ = [
    "MaxDivSolverBuilder",
    "ParallelMaxDivSolverBuilder",
    "SolverBuilderBase",
]
