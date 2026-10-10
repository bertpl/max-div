"""This module holds the validation steps: the release's preconditions, checked before anything is written."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from helpers import (
    CHANGELOG,
    PACKAGE_NAME,
    PYPROJECT,
    PYTHON_VERSIONS_FILE,
    UNRELEASED_SECTION_RE,
    fail_with_message,
    parse_semver,
    run_command,
)
from step import BadgeMetrics, ReleaseContext, ReleasePhase, ReleaseStep

# warn if the cumulative union exceeds this multiple of the largest single combo
TEST_COUNT_UNION_RATIO_WARN = 1.5

# how long to wait for an in-flight 'Push to Main' run on HEAD, and how often to re-check
CI_WAIT_TIMEOUT_SEC = 25 * 60
CI_POLL_INTERVAL_SEC = 30


# ==================================================================================================
#  ValidationPhaseStep
# ==================================================================================================
class ValidationPhaseStep(ReleaseStep):
    """A validation step checks a precondition of the release and writes nothing to the repo."""

    phase = ReleasePhase.VALIDATION


# ==================================================================================================
#  Steps
# ==================================================================================================
class CheckCleanWorkingTreeOnMainStep(ValidationPhaseStep):
    """Validate working tree is on main and clean."""

    def title(self, context: ReleaseContext) -> str:
        return "working tree on main and clean"

    def run(self, context: ReleaseContext) -> None:
        branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
        if branch != "main":
            fail_with_message(f"not on main (currently on {branch})")
        porcelain = run_command(["git", "status", "--porcelain"])
        if porcelain.strip():
            fail_with_message("working tree has uncommitted changes:\n" + porcelain)


class CheckMainInSyncWithOriginStep(ValidationPhaseStep):
    """Validate main is in sync with origin."""

    def title(self, context: ReleaseContext) -> str:
        return "main in sync with origin"

    def run(self, context: ReleaseContext) -> None:
        run_command(["git", "fetch", "origin", "main"])
        local = run_command(["git", "rev-parse", "HEAD"]).strip()
        remote = run_command(["git", "rev-parse", "origin/main"]).strip()
        if local != remote:
            fail_with_message(f"local main ({local[:8]}) does not match origin/main ({remote[:8]})")


class CheckVersionUpgradeStep(ValidationPhaseStep):
    """Validate VERSION is strictly greater than current."""

    def title(self, context: ReleaseContext) -> str:
        return f"VERSION {context.version} is an upgrade"

    def run(self, context: ReleaseContext) -> None:
        new = parse_semver(context.version)
        current_version = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
        if new <= parse_semver(current_version):
            fail_with_message(f"VERSION {context.version} is not greater than current {current_version}")


class CheckTagDoesNotExistStep(ValidationPhaseStep):
    """Validate tag does not exist locally or on origin."""

    def title(self, context: ReleaseContext) -> str:
        return f"tag v{context.version} does not exist (local + remote)"

    def run(self, context: ReleaseContext) -> None:
        tag = f"v{context.version}"
        if run_command(["git", "tag", "-l", tag]).strip():
            fail_with_message(f"tag {tag} already exists locally")
        if run_command(["git", "ls-remote", "--tags", "origin", tag]).strip():
            fail_with_message(f"tag {tag} already exists on origin")


class CheckVersionNotOnPyPIStep(ValidationPhaseStep):
    """Validate version is not already on PyPI."""

    def title(self, context: ReleaseContext) -> str:
        return f"version {context.version} is not on PyPI"

    def run(self, context: ReleaseContext) -> None:
        url = f"https://pypi.org/pypi/{PACKAGE_NAME}/{context.version}/json"
        try:
            with urllib.request.urlopen(url, timeout=10):
                fail_with_message(f"version {context.version} is already published on PyPI")
        except urllib.error.HTTPError as e:
            # 404 = not published yet (the good case); any other status is a check failure.
            if e.code != 404:
                fail_with_message(f"PyPI check returned HTTP {e.code}")
        except urllib.error.URLError as e:
            # HTTPError is a subclass of URLError, so this only catches transport failures
            # (DNS, connection reset, the 10s timeout) — exit with a one-line error message, not a raw traceback.
            fail_with_message(f"could not reach PyPI to check {context.version}: {e.reason}")


class CheckClassifiersMatchPythonVersionsStep(ValidationPhaseStep):
    """Validate Python classifiers match .python-versions."""

    def title(self, context: ReleaseContext) -> str:
        return "Python classifiers in pyproject.toml match .python-versions"

    def run(self, context: ReleaseContext) -> None:
        versions = [v.strip() for v in PYTHON_VERSIONS_FILE.read_text().split() if v.strip()]
        text = PYPROJECT.read_text()
        declared = set(re.findall(r'"Programming Language :: Python :: ([\d.]+)"', text))
        expected = set(versions)
        missing = expected - declared
        extra = declared - expected
        if missing or extra:
            fail_with_message(
                f"classifiers do not match .python-versions. "
                f"Missing: {sorted(missing) or 'none'}; "
                f"Extra: {sorted(extra) or 'none'}"
            )


class CheckChangelogHasEntriesStep(ValidationPhaseStep):
    """Validate Unreleased section has at least one bullet entry."""

    def title(self, context: ReleaseContext) -> str:
        return "CHANGELOG.md '## Unreleased' has at least one entry"

    def run(self, context: ReleaseContext) -> None:
        m = UNRELEASED_SECTION_RE.search(CHANGELOG.read_text())
        if not m:
            fail_with_message("no '## Unreleased' section in CHANGELOG.md")
        if not re.search(r"^- ", m.group(1), re.MULTILINE):
            fail_with_message("'## Unreleased' has no bullet entries")


class GatherBadgeMetricsStep(ValidationPhaseStep):
    """Resolve every badge number, failing the release if any of them cannot be obtained.

    When the 'Push to Main' run for HEAD is still running, the step waits for it. After a last-minute commit (e.g.
    a changelog entry), CI has usually not finished yet, and waiting saves the maintainer from watching CI and
    rerunning the release by hand.
    """

    def title(self, context: ReleaseContext) -> str:
        return "gather badge metrics (CI metrics for HEAD)"

    def run(self, context: ReleaseContext) -> None:
        metrics = self._fetch_release_metrics()
        union = int(metrics["test_union"])
        max_combo = int(metrics["test_max"])
        if max_combo and union > TEST_COUNT_UNION_RATIO_WARN * max_combo:
            print(
                f"\nWARNING: cumulative test count ({union}) exceeds "
                f"{TEST_COUNT_UNION_RATIO_WARN}x the largest single combo ({max_combo}). "
                "Node-id mismatches across combos can inflate the union — verify before publishing.\n",
                file=sys.stderr,
            )
        context.badge_metrics = BadgeMetrics(coverage_pct=float(metrics["coverage_pct"]), test_union=union)

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @classmethod
    def _fetch_release_metrics(cls) -> dict[str, float]:
        """Download CI's cumulative metrics for the commit being released.

        The numbers come from the matrix combine job, not a local run, so the
        badge matches the CI gate exactly. It waits up to `CI_WAIT_TIMEOUT_SEC` for HEAD's
        'Push to Main' run to succeed, and exits when no such run succeeds in time.
        """
        head_sha = run_command(["git", "rev-parse", "HEAD"]).strip()
        run_id = cls._wait_for_main_ci(head_sha)
        with tempfile.TemporaryDirectory() as tmp:
            run_command(["gh", "run", "download", run_id, "--name", "release-metrics", "--dir", tmp])
            return json.loads((Path(tmp) / "metrics.json").read_text())

    @classmethod
    def _wait_for_main_ci(cls, head_sha: str) -> str:
        """Return the id of a successful 'Push to Main' run for `head_sha`, waiting one out if in flight.

        Aborts when:

        - no run exists for `head_sha` (the push did not trigger CI);
        - the run concluded without success;
        - `CI_WAIT_TIMEOUT_SEC` passes without the run completing.

        Once found, the run is polled by its id, so an out-of-date result from `gh run list` cannot
        hide the run.
        """
        run_id = cls._main_run_id_for(head_sha)
        if run_id is None:
            fail_with_message(f"no 'Push to Main' run found for HEAD {head_sha[:8]} — did the push trigger CI?")
        deadline = time.monotonic() + CI_WAIT_TIMEOUT_SEC
        has_announced_wait = False
        while True:
            state = json.loads(run_command(["gh", "run", "view", run_id, "--json", "status,conclusion"]))
            # the conclusion is empty until the run completes
            status, conclusion = state["status"], state.get("conclusion") or ""
            if status == "completed":
                if conclusion != "success":
                    fail_with_message(f"'Push to Main' run for HEAD {head_sha[:8]} concluded '{conclusion}'")
                return run_id
            if not has_announced_wait:
                print(f"       CI for HEAD {head_sha[:8]} is {status} — waiting up to {CI_WAIT_TIMEOUT_SEC // 60} min")
                has_announced_wait = True
            if time.monotonic() >= deadline:
                fail_with_message(f"timed out after {CI_WAIT_TIMEOUT_SEC // 60} min waiting for CI on {head_sha[:8]}")
            time.sleep(CI_POLL_INTERVAL_SEC)

    @staticmethod
    def _main_run_id_for(head_sha: str) -> str | None:
        """Return the id of the newest 'Push to Main' run for commit `head_sha`, or None if it has none.

        The lookup filters by commit: the unfiltered `gh run list` on `main` can briefly return an
        out-of-date result, which leaves out a run that exists.
        """
        out = run_command(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                "push_to_main.yml",
                "--commit",
                head_sha,
                "--limit",
                "1",
                "--json",
                "databaseId",
            ]
        )
        runs = json.loads(out)
        if runs:
            return str(runs[0]["databaseId"])
        else:
            return None
