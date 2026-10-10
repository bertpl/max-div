"""This module runs the release: it holds the list of release steps and the loop that runs them in order."""

from __future__ import annotations

import argparse
import subprocess

from helpers import PACKAGE_NAME
from phase_post_release import AddUnreleasedSectionStep, CommitNextCycleStep, PushMainAndTagStep
from phase_release_commit import (
    BumpVersionStep,
    CreateTagStep,
    FinalizeChangelogStep,
    RefreshUvLockStep,
    StampReadmeAndSplashThenCommitStep,
)
from phase_validation import (
    CheckChangelogHasEntriesStep,
    CheckClassifiersMatchPythonVersionsStep,
    CheckCleanWorkingTreeOnMainStep,
    CheckMainInSyncWithOriginStep,
    CheckTagDoesNotExistStep,
    CheckVersionNotOnPyPIStep,
    CheckVersionUpgradeStep,
    GatherBadgeMetricsStep,
)
from step import ReleaseContext, ReleasePhase, ReleaseStep

# The release runs these steps in this order and numbers them by their position, so a step is added
# or moved by editing this list alone. The order has 2 constraints:
# - the steps of each phase stay together, in the order that ReleasePhase lists the phases;
# - a step that reads a ReleaseContext field comes after the step that sets it.
RELEASE_STEPS: list[ReleaseStep] = [
    CheckCleanWorkingTreeOnMainStep(),
    CheckMainInSyncWithOriginStep(),
    CheckVersionUpgradeStep(),
    CheckTagDoesNotExistStep(),
    CheckVersionNotOnPyPIStep(),
    CheckClassifiersMatchPythonVersionsStep(),
    CheckChangelogHasEntriesStep(),
    # GatherBadgeMetricsStep is the last precondition: it downloads the CI metrics after every cheap
    # check has passed and before the first write.
    GatherBadgeMetricsStep(),
    BumpVersionStep(),
    RefreshUvLockStep(),
    FinalizeChangelogStep(),
    StampReadmeAndSplashThenCommitStep(),
    CreateTagStep(),
    AddUnreleasedSectionStep(),
    CommitNextCycleStep(),
    PushMainAndTagStep(),
]


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the release steps."""
    parser = argparse.ArgumentParser(description="Release max-div: validate, commit, tag and push version X.Y.Z.")
    parser.add_argument("version", help="X.Y.Z (no leading v)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every precondition, including the CI badge-metrics fetch, then stop before the first write",
    )
    args = parser.parse_args(argv)
    context = ReleaseContext(args.version)

    print(f"Releasing {PACKAGE_NAME} v{args.version}")
    run_release(RELEASE_STEPS, context, is_dry_run=args.dry_run)


def run_release(steps: list[ReleaseStep], context: ReleaseContext, is_dry_run: bool) -> None:
    """Run `steps` in order, numbered by position, printing each phase's name as it starts.

    A dry run runs only the validation steps and then prints the badge metrics, so one of those steps
    must set `context.badge_metrics`. When a step raises `SystemExit` or `CalledProcessError`, its
    `on_failure` runs before the exception propagates; any other exception propagates without it.
    """
    if is_dry_run:
        steps = [step for step in steps if step.phase is ReleasePhase.VALIDATION]
    phase = None
    for number, step in enumerate(steps, start=1):
        if step.phase is not phase:
            phase = step.phase
            print(f"\n{phase.value}:")
        print(f"  [{number:>2}] {step.title(context)}")
        try:
            step.run(context)
        except (subprocess.CalledProcessError, SystemExit):
            step.on_failure(context)
            raise

    if is_dry_run:
        badge_metrics = context.badge_metrics
        print(
            f"\nDry run: every precondition passed and nothing was written.\n"
            f"  coverage {badge_metrics.coverage_pct:.2f}% | tests {badge_metrics.test_union}\n"
        )
