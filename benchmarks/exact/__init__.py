"""Exact-solver references: formulations that prove optimality at small problem sizes."""

from .cpsat_maxmin import CpsatMaxMinResult, solve_maxmin_cpsat
from .cpsat_nn_assignment import CpsatNnAssignmentResult, solve_nn_assignment_cpsat
from .mip_maxmin import MaxMinResult, solve_maxmin_highs, solve_maxmin_scip
from .scip_nn_separation import ScipNnSeparationResult, solve_nn_separation_scip

__all__ = [
    "CpsatMaxMinResult",
    "CpsatNnAssignmentResult",
    "MaxMinResult",
    "ScipNnSeparationResult",
    "solve_maxmin_cpsat",
    "solve_maxmin_highs",
    "solve_maxmin_scip",
    "solve_nn_assignment_cpsat",
    "solve_nn_separation_scip",
]
