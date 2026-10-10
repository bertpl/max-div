"""This module defines what a release step is: its phase, the shared state of a release, and the base class."""

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

    phase: ClassVar[ReleasePhase]

    @abstractmethod
    def title(self, context: ReleaseContext) -> str:
        """Return this step's title, printed when the step starts."""

    @abstractmethod
    def run(self, context: ReleaseContext) -> None:
        """Do the step.

        On failure, call `fail_with_message`, or let the `CalledProcessError` of `run_command` propagate: those are
        the 2 failures that `run_release` handles.
        """

    # Not abstract: doing nothing is the default, and only a step whose failure leaves something to undo overrides it.
    def on_failure(self, context: ReleaseContext) -> None:  # noqa: B027
        """Run after `run` fails, before the failure propagates; the default does nothing."""


# ==================================================================================================
#  ReleasePhase
# ==================================================================================================
class ReleasePhase(Enum):
    """A phase of the release; the phases run in the order that they are listed here."""

    # This phase only checks: its steps write nothing to the repo, and `--dry-run` stops after this phase.
    VALIDATION = "Validation"
    # This phase builds the release commit and its tag, locally; nothing is pushed yet.
    RELEASE_COMMIT = "Release commit"
    # This phase runs after the tag exists, so a failure here leaves a local release commit and tag to undo.
    POST_RELEASE = "Post-release"


# ==================================================================================================
#  ReleaseContext
# ==================================================================================================
@dataclass
class ReleaseContext:
    """A ReleaseContext holds the state of 1 release, shared by all its steps.

    It carries the version being released, and the results that an earlier step stores for a later step to read,
    such as the badge metrics.
    """

    version: str
    badge_metrics: BadgeMetrics | None = None


@dataclass(frozen=True)
class BadgeMetrics:
    """A BadgeMetrics records the badge numbers for one release.

    They are CI's combined coverage percentage, and `test_union`, the number of distinct tests across all CI matrix
    combos.
    """

    coverage_pct: float
    test_union: int

    @property
    def coverage_color(self) -> str:
        """Return the shields.io badge color for `coverage_pct`."""
        if self.coverage_pct >= 90:
            return "brightgreen"
        elif self.coverage_pct >= 75:
            return "yellow"
        else:
            return "red"
