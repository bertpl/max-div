"""Runners: execute max-div over a budget series, or a single-shot adapter, emitting run records."""

from .adapter_runner import run_adapter
from .maxdiv_runner import run_maxdiv_budget_series

__all__ = [
    "run_adapter",
    "run_maxdiv_budget_series",
]
