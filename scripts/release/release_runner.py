"""This module runs the release: it holds the list of release steps and the loop that runs them in order."""

from __future__ import annotations

import argparse
import subprocess
import sys

from post_release import AddUnreleasedSection, CommitNextCycle, PushMainAndTag
from release_commit import BumpVersion, CommitRelease, CreateTag, FinalizeChangelog, RefreshLock
from release_helpers import PACKAGE_NAME, parse_semver
from release_step import Phase, ReleaseContext, ReleaseStep
from validation import (
    CheckChangelogHasEntries,
    CheckClassifiers,
    CheckInSync,
    CheckNotOnPyPI,
    CheckTagDoesNotExist,
    CheckVersionUpgrade,
    CheckWorkingTree,
    GatherBadgeMetrics,
)

# The release runs these steps in this order and numbers them by their position, so a step is
# added or moved by editing this list alone.
RELEASE_STEPS: list[ReleaseStep] = [
    CheckWorkingTree(),
    CheckInSync(),
    CheckVersionUpgrade(),
    CheckTagDoesNotExist(),
    CheckNotOnPyPI(),
    CheckClassifiers(),
    CheckChangelogHasEntries(),
    GatherBadgeMetrics(),
    BumpVersion(),
    RefreshLock(),
    FinalizeChangelog(),
    CommitRelease(),
    CreateTag(),
    AddUnreleasedSection(),
    CommitNextCycle(),
    PushMainAndTag(),
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
    parse_semver(args.version)

    print(f"Releasing {PACKAGE_NAME} v{args.version}")
    run_release(RELEASE_STEPS, ReleaseContext(args.version), is_dry_run=args.dry_run)


def run_release(steps: list[ReleaseStep], context: ReleaseContext, is_dry_run: bool) -> None:
    """Run `steps` in order, numbered by position, printing each phase's name as it starts.

    A dry run runs only the validation steps. A failure in a post-release step, when the release
    commit and the tag exist but are not pushed, prints how to undo them.
    """
    if is_dry_run:
        steps = [step for step in steps if step.phase is Phase.VALIDATION]
    phase = None
    for number, step in enumerate(steps, start=1):
        if step.phase is not phase:
            phase = step.phase
            print(f"\n{phase.value}:")
        print(f"  [{number:>2}] {step.title(context)}")
        try:
            step.run(context)
        except (subprocess.CalledProcessError, SystemExit):
            if step.phase is Phase.POST_RELEASE:
                _print_recovery_hint(context.version)
                sys.exit(1)
            else:
                raise

    if is_dry_run:
        badges = context.badge_metrics
        print(
            f"\nDry run: every precondition passed and nothing was written.\n"
            f"  coverage {badges.coverage_pct:.2f}% | tests {badges.test_union}\n"
        )


def _print_recovery_hint(version: str) -> None:
    """Print how to undo the local release commit, the next-cycle commit if any, and the tag.

    The tag points at the release commit, so the commit before the tag is where `main` stood
    before the release, whichever post-release step failed.
    """
    print(
        f"\nERROR: a post-release step failed.\n"
        f"Local state: release commit and tag v{version} created, not pushed.\n"
        f"To abort and retry:\n"
        f"  git reset --hard v{version}~1\n"
        f"  git tag -d v{version}\n",
        file=sys.stderr,
    )
