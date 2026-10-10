"""This module holds the post-release steps: open the next development cycle, then push main and the tag."""

from __future__ import annotations

import re
import sys

from release_helpers import CATEGORIES, CHANGELOG, fail_with_message, run_command
from release_step import ReleaseContext, ReleasePhase, ReleaseStep


# ==================================================================================================
#  PostReleasePhaseStep
# ==================================================================================================
class PostReleasePhaseStep(ReleaseStep):
    """A post-release step runs after the tag exists; its failure leaves a local release commit and tag."""

    phase = ReleasePhase.POST_RELEASE

    def on_failure(self, context: ReleaseContext) -> None:
        """Print how to undo the local release commit and tag, then exit.

        The tag points at the release commit, so the commit before the tag is where `main` stood before the
        release, no matter which post-release step failed. Resetting to that commit also drops the
        'chore: begin next development cycle' commit if it exists.
        """
        print(
            f"\nERROR: a post-release step failed.\n"
            f"Local state: release commit and tag v{context.version} created, not pushed.\n"
            f"To abort and retry:\n"
            f"  git reset --hard v{context.version}~1\n"
            f"  git tag -d v{context.version}\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ==================================================================================================
#  Steps
# ==================================================================================================
class AddUnreleasedSectionStep(PostReleasePhaseStep):
    """Add a fresh Unreleased section to the changelog."""

    def title(self, context: ReleaseContext) -> str:
        return "add fresh '## Unreleased' section to CHANGELOG.md"

    def run(self, context: ReleaseContext) -> None:
        text = CHANGELOG.read_text()
        m = re.search(r"^## ", text, re.MULTILINE)
        if not m:
            fail_with_message("CHANGELOG.md has no version sections")
        insertion = "## Unreleased\n\n" + "\n".join(f"### {c}\n" for c in CATEGORIES) + "\n"
        text = text[: m.start()] + insertion + text[m.start() :]
        CHANGELOG.write_text(text)


class CommitNextCycleStep(PostReleasePhaseStep):
    """Commit the fresh Unreleased section."""

    def title(self, context: ReleaseContext) -> str:
        return "commit 'chore: begin next development cycle'"

    def run(self, context: ReleaseContext) -> None:
        run_command(["git", "add", "CHANGELOG.md"])
        run_command(["git", "commit", "-m", "chore: begin next development cycle"])


class PushMainAndTagStep(PostReleasePhaseStep):
    """Push main and the tag atomically."""

    def title(self, context: ReleaseContext) -> str:
        return f"push main + v{context.version} atomically"

    def run(self, context: ReleaseContext) -> None:
        run_command(["git", "push", "--atomic", "origin", "main", f"refs/tags/v{context.version}"])
