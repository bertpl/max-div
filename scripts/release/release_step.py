"""This module defines what a release step is: its phase, the state that the steps share, and the base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


# ==================================================================================================
#  ReleaseStep
# ==================================================================================================
class ReleaseStep(ABC):
    """A release step is 1 numbered action of the release: a check, or a change to the repo."""

    phase: ClassVar[Phase]

    @abstractmethod
    def title(self, context: ReleaseContext) -> str:
        """Return the line that the release prints when this step starts."""

    @abstractmethod
    def run(self, context: ReleaseContext) -> None:
        """Do the step; exit the release with an error message if it fails."""


# ==================================================================================================
#  Phase
# ==================================================================================================
class Phase(Enum):
    """A phase of the release; the phases run in the order that they are listed here."""

    # Checks only: a step of this phase writes nothing to the repo, and `--dry-run` stops after it.
    VALIDATION = "Validation"
    # Bumps the version, finalizes the changelog, commits the release and tags it.
    RELEASE_COMMIT = "Release commit"
    # Runs after the tag exists, so a failure here leaves a local release commit and tag to undo.
    POST_RELEASE = "Post-release"


# ==================================================================================================
#  ReleaseContext
# ==================================================================================================
@dataclass
class ReleaseContext:
    """A ReleaseContext holds the state that the steps of 1 release share.

    It carries the version being released, and what an earlier step finds for a later one.
    """

    version: str
    badge_metrics: BadgeMetrics | None = None


@dataclass(frozen=True)
class BadgeMetrics:
    """A BadgeMetrics records the badge numbers for one release."""

    coverage_pct: float
    test_union: int
