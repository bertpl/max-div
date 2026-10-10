"""This module holds the post-release steps: open the next development cycle, then push main and the tag."""

from __future__ import annotations

import re

from release_helpers import CATEGORIES, CHANGELOG, fail_with_message, run_command
from release_step import Phase, ReleaseContext, ReleaseStep


# ==================================================================================================
#  PostReleaseStep
# ==================================================================================================
class PostReleaseStep(ReleaseStep):
    """A post-release step runs after the tag exists; its failure leaves a local release commit and tag."""

    phase = Phase.POST_RELEASE


# ==================================================================================================
#  Steps
# ==================================================================================================
class AddUnreleasedSection(PostReleaseStep):
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


class CommitNextCycle(PostReleaseStep):
    """Commit the fresh Unreleased section."""

    def title(self, context: ReleaseContext) -> str:
        return "commit 'chore: begin next development cycle'"

    def run(self, context: ReleaseContext) -> None:
        run_command(["git", "add", "CHANGELOG.md"])
        run_command(["git", "commit", "-m", "chore: begin next development cycle"])


class PushMainAndTag(PostReleaseStep):
    """Push main and the tag atomically."""

    def title(self, context: ReleaseContext) -> str:
        return f"push main + v{context.version} atomically"

    def run(self, context: ReleaseContext) -> None:
        run_command(["git", "push", "--atomic", "origin", "main", f"refs/tags/v{context.version}"])
