"""Separation-family diversity contribution: one backend per storage layout, plus the tracker."""

from ._backends import SeparationBackend, backend_for
from ._tracker import SeparationTracker

__all__ = ["SeparationBackend", "SeparationTracker", "backend_for"]
