"""Exact-solver references: formulations that prove optimality at small problem sizes."""

from .cpsat_maxmin import CpsatMaxMinResult, solve_maxmin_cpsat

__all__ = [
    "CpsatMaxMinResult",
    "solve_maxmin_cpsat",
]
