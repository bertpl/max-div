"""This module holds the release commit steps, which build the release commit and its tag locally."""

from __future__ import annotations

import re
import shutil
from datetime import date

from helpers import (
    CATEGORIES,
    CHANGELOG,
    README,
    REPO_ROOT,
    SPLASH_SCRIPT,
    SPLASH_WEBP,
    UNRELEASED_SECTION_RE,
    fail_with_message,
    run_command,
)
from step import BadgeMetrics, ReleaseContext, ReleasePhase, ReleaseStep


# ==================================================================================================
#  ReleaseCommitPhaseStep
# ==================================================================================================
class ReleaseCommitPhaseStep(ReleaseStep):
    """A release commit step builds the release commit or its tag, locally; nothing is pushed yet."""

    phase = ReleasePhase.RELEASE_COMMIT


# ==================================================================================================
#  Steps
# ==================================================================================================
class BumpVersionStep(ReleaseCommitPhaseStep):
    """Set version in pyproject.toml."""

    def title(self, context: ReleaseContext) -> str:
        return f"bump version to {context.version}"

    def run(self, context: ReleaseContext) -> None:
        run_command(["uv", "version", context.version])


class RefreshUvLockStep(ReleaseCommitPhaseStep):
    """Refresh uv.lock after version bump."""

    def title(self, context: ReleaseContext) -> str:
        return "refresh uv.lock"

    def run(self, context: ReleaseContext) -> None:
        run_command(["uv", "lock"])


class FinalizeChangelogStep(ReleaseCommitPhaseStep):
    """Turn the Unreleased section into a dated version section, dropping the categories that have no entries."""

    def title(self, context: ReleaseContext) -> str:
        return f"finalize CHANGELOG.md '## Unreleased' -> '## {context.version} ({date.today().isoformat()})'"

    def run(self, context: ReleaseContext) -> None:
        text = CHANGELOG.read_text()
        m = UNRELEASED_SECTION_RE.search(text)
        if not m:
            fail_with_message("no '## Unreleased' section to finalize")
        body = m.group(1)
        new_body_lines: list[str] = []
        lines = body.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i]
            cat_match = re.match(r"^### (\w+)\s*$", line)
            if cat_match and cat_match.group(1) in CATEGORIES:
                j = i + 1
                has_entry = False
                while j < len(lines) and not re.match(r"^### ", lines[j]):
                    if lines[j].lstrip().startswith("- "):
                        has_entry = True
                        break
                    j += 1
                if has_entry:
                    new_body_lines.append(line)
                    i += 1
                    while i < len(lines) and not re.match(r"^### ", lines[i]):
                        new_body_lines.append(lines[i])
                        i += 1
                else:
                    i += 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
            else:
                new_body_lines.append(line)
                i += 1
        tail = text[m.end() :]
        # When a release section follows, keep a blank line before its `## ` heading, so the finalized
        # section's last bullet is not left directly above that heading.
        new_body = "".join(new_body_lines).rstrip() + ("\n\n" if tail else "\n")
        new_header = f"## {context.version} ({date.today().isoformat()})\n"
        text = text[: m.start()] + new_header + new_body + tail
        CHANGELOG.write_text(text)


class StampReadmeAndSplashThenCommitStep(ReleaseCommitPhaseStep):
    """Refresh the README badges from `context.badge_metrics`, stamp the splash, then create the release commit.

    `GatherBadgeMetricsStep` sets `context.badge_metrics`, so it must run before this step.
    """

    def title(self, context: ReleaseContext) -> str:
        return f"refresh README badges + stamp splash + commit 'release: {context.version}'"

    def run(self, context: ReleaseContext) -> None:
        if context.badge_metrics is None:
            fail_with_message("the badge metrics were not gathered before the release commit")
        self._refresh_readme_badges(context.badge_metrics)
        self._stamp_readme_splash_url(context.version)
        self._stamp_splash(context.version)
        run_command(["git", "add", "pyproject.toml", "uv.lock", "CHANGELOG.md", "README.md", str(SPLASH_WEBP)])
        run_command(["git", "commit", "-m", f"release: {context.version}"])

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _refresh_readme_badges(badge_metrics: BadgeMetrics) -> None:
        """Stamp the README coverage + test-count badges from `badge_metrics`."""
        text = README.read_text()
        text = re.sub(
            r"badge/coverage-[\d.]+%25-[a-z]+",
            f"badge/coverage-{badge_metrics.coverage_pct:.2f}%25-{badge_metrics.coverage_color}",
            text,
        )
        text = re.sub(r"badge/tests-\d+-blue", f"badge/tests-{badge_metrics.test_union}-blue", text)
        README.write_text(text)

    @staticmethod
    def _stamp_readme_splash_url(version: str) -> None:
        """Pin the README splash image URL to the release tag.

        The README references the splash via a raw.githubusercontent.com URL pinned to a tag,
        so GitHub, PyPI, and Read the Docs all render the splash belonging to their version.
        """
        text = README.read_text()
        text = re.sub(
            r"raw\.githubusercontent\.com/bertpl/max-div/v[\d.]+/images/splash_with_version\.webp",
            f"raw.githubusercontent.com/bertpl/max-div/v{version}/images/splash_with_version.webp",
            text,
        )
        README.write_text(text)

    @staticmethod
    def _stamp_splash(version: str) -> None:
        """Stamp the release version onto the committed splash webp (needs ImageMagick).

        It runs the version-overlay stage of ``create_splash_with_version.sh`` on the committed,
        version-independent base image. It exits with an error if ``magick`` is absent, since a
        maintainer-driven release must produce the real asset.
        """
        if shutil.which("magick") is None:
            fail_with_message("ImageMagick ('magick') is required to stamp the release splash but was not found")
        run_command(["sh", str(SPLASH_SCRIPT), f"v{version}"], cwd=REPO_ROOT)


class CreateTagStep(ReleaseCommitPhaseStep):
    """Create the version tag."""

    def title(self, context: ReleaseContext) -> str:
        return f"create tag v{context.version}"

    def run(self, context: ReleaseContext) -> None:
        run_command(["git", "tag", f"v{context.version}"])
