"""Adapters: one per compared tool/baseline, all implementing SelectionAdapter."""

from .base import SelectionAdapter
from .random_baseline import RandomBaseline

__all__ = [
    "RandomBaseline",
    "SelectionAdapter",
]
