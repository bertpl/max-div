"""Two builders configure a solve: one builds a single solver, the other a parallel solver running several workers."""

from ._base import SolverBuilderBase
from ._parallel import ParallelMaxDivSolverBuilder
from ._single import MaxDivSolverBuilder

__all__ = [
    "MaxDivSolverBuilder",
    "ParallelMaxDivSolverBuilder",
    "SolverBuilderBase",
]
