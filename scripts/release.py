"""Release driver for max-div.

Run via ``make release VERSION=X.Y.Z``.  Validates state, bumps version, stamps
the versioned splash + README badges, finalizes the changelog, commits, tags,
opens a fresh Unreleased section, and pushes main + tag atomically.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
README = REPO_ROOT / "README.md"
PYTHON_VERSIONS_FILE = REPO_ROOT / ".python-versions"
SPLASH_SCRIPT = REPO_ROOT / "images" / "splash" / "create_splash_with_version.sh"
SPLASH_WEBP = REPO_ROOT / "images" / "splash_with_version.webp"

PACKAGE_NAME = "max-div"
CATEGORIES = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ==================================================================================================
#  helpers
# ==================================================================================================
def run_command(cmd: list[str], **kw: object) -> str:
    """Run a subprocess and return stdout, or exit on failure."""
    # check=False is deliberate: the return code is handled below with
    # richer diagnostics than subprocess's own CalledProcessError.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)
    if result.returncode != 0:
        sys.stderr.write(f"\n$ {' '.join(cmd)}\n")
        if result.stdout:
            sys.stderr.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.stdout


def print_step(n: int, msg: str) -> None:
    """Print a numbered step message."""
    print(f"  [{n:>2}] {msg}")


def fail_with_message(msg: str, code: int = 1) -> None:
    """Print an error and exit."""
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse and validate a semver string."""
    if not SEMVER_RE.match(version):
        fail_with_message(f"VERSION {version!r} is not in X.Y.Z form")
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def read_pyproject_version() -> str:
    """Read the current version from pyproject.toml."""
    text = PYPROJECT.read_text()
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if not m:
        fail_with_message("Could not find version in pyproject.toml")
    return m.group(1)


def read_python_versions() -> list[str]:
    """Read supported Python versions from .python-versions."""
    return [v.strip() for v in PYTHON_VERSIONS_FILE.read_text().split() if v.strip()]


# ==================================================================================================
#  validation steps (1-7)
# ==================================================================================================
def step_1_check_working_tree() -> None:
    """Validate working tree is on main and clean."""
    print_step(1, "working tree on main and clean")
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if branch != "main":
        fail_with_message(f"not on main (currently on {branch})")
    porcelain = run_command(["git", "status", "--porcelain"])
    if porcelain.strip():
        fail_with_message("working tree has uncommitted changes:\n" + porcelain)


def step_2_check_in_sync() -> None:
    """Validate main is in sync with origin."""
    print_step(2, "main in sync with origin")
    run_command(["git", "fetch", "origin", "main"])
    local = run_command(["git", "rev-parse", "HEAD"]).strip()
    remote = run_command(["git", "rev-parse", "origin/main"]).strip()
    if local != remote:
        fail_with_message(f"local main ({local[:8]}) does not match origin/main ({remote[:8]})")


def step_3_check_version_upgrade(version: str) -> None:
    """Validate VERSION is strictly greater than current."""
    print_step(3, f"VERSION {version} is an upgrade")
    new = parse_semver(version)
    current = parse_semver(read_pyproject_version())
    if new <= current:
        fail_with_message(f"VERSION {version} is not greater than current {'.'.join(str(p) for p in current)}")


def step_4_check_tag_doesnt_exist(version: str) -> None:
    """Validate tag does not exist locally or on origin."""
    print_step(4, f"tag v{version} does not exist (local + remote)")
    tag = f"v{version}"
    if run_command(["git", "tag", "-l", tag]).strip():
        fail_with_message(f"tag {tag} already exists locally")
    if run_command(["git", "ls-remote", "--tags", "origin", tag]).strip():
        fail_with_message(f"tag {tag} already exists on origin")


def step_5_check_pypi_doesnt_have(version: str) -> None:
    """Validate version is not already on PyPI."""
    print_step(5, f"version {version} is not on PyPI")
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=10):
            fail_with_message(f"version {version} is already published on PyPI")
    except urllib.error.HTTPError as e:
        # 404 = not published yet (the good case); any other status is a check failure.
        if e.code != 404:
            fail_with_message(f"PyPI check returned HTTP {e.code}")
    except urllib.error.URLError as e:
        # HTTPError is a subclass of URLError, so this only catches transport failures
        # (DNS, connection reset, the 10s timeout) — fail cleanly instead of a raw traceback.
        fail_with_message(f"could not reach PyPI to check {version}: {e.reason}")


def step_6_check_classifiers_match() -> None:
    """Validate Python classifiers match .python-versions."""
    print_step(6, "Python classifiers in pyproject.toml match .python-versions")
    versions = read_python_versions()
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


def step_7_check_changelog_has_entries() -> None:
    """Validate Unreleased section has at least one bullet entry."""
    print_step(7, "CHANGELOG.md '## Unreleased' has at least one entry")
    text = CHANGELOG.read_text()
    m = re.search(r"^## Unreleased\s*$(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    if not m:
        fail_with_message("no '## Unreleased' section in CHANGELOG.md")
    if not re.search(r"^- ", m.group(1), re.MULTILINE):
        fail_with_message("'## Unreleased' has no bullet entries")


# ==================================================================================================
#  release commit steps (8-12)
# ==================================================================================================
def step_8_bump_version(version: str) -> None:
    """Set version in pyproject.toml."""
    print_step(8, f"bump version to {version}")
    run_command(["uv", "version", version])


def step_9_lock() -> None:
    """Refresh uv.lock after version bump."""
    print_step(9, "refresh uv.lock")
    run_command(["uv", "lock"])


def step_10_finalize_changelog(version: str) -> None:
    """Move Unreleased entries to a dated version section."""
    print_step(10, f"finalize CHANGELOG.md '## Unreleased' -> '## {version} ({date.today().isoformat()})'")
    text = CHANGELOG.read_text()
    m = re.search(r"^## Unreleased\s*$(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
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
    # section's last bullet is not left directly above it.
    new_body = "".join(new_body_lines).rstrip() + ("\n\n" if tail else "\n")
    new_header = f"## {version} ({date.today().isoformat()})\n"
    text = text[: m.start()] + new_header + new_body + tail
    CHANGELOG.write_text(text)


# warn if the cumulative union exceeds this multiple of the largest single combo
TEST_COUNT_UNION_RATIO_WARN = 1.5


def _latest_main_coverage_run() -> tuple[str, str]:
    """Return (run_id, head_sha) of the latest successful 'Push to Main' run."""
    out = run_command(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            "push_to_main.yml",
            "--branch",
            "main",
            "--status",
            "success",
            "--limit",
            "1",
            "--json",
            "databaseId,headSha",
        ]
    )
    runs = json.loads(out)
    if not runs:
        fail_with_message("no successful 'Push to Main' run found to source coverage metrics from")
    return str(runs[0]["databaseId"]), runs[0]["headSha"]


def _fetch_release_metrics() -> dict[str, float]:
    """Download CI's cumulative metrics for the commit being released.

    The numbers come from the matrix combine job, not a local run, so the
    badge matches the CI gate exactly. Fails if the latest green main run is
    not the commit at HEAD (i.e. CI on current main hasn't gone green yet).
    """
    run_id, head_sha = _latest_main_coverage_run()
    local_head = run_command(["git", "rev-parse", "HEAD"]).strip()
    if head_sha != local_head:
        fail_with_message(
            f"latest main coverage run is for {head_sha[:8]}, not HEAD {local_head[:8]} — "
            "wait for CI on current main to go green before releasing"
        )
    with tempfile.TemporaryDirectory() as tmp:
        run_command(["gh", "run", "download", run_id, "--name", "release-metrics", "--dir", tmp])
        return json.loads((Path(tmp) / "metrics.json").read_text())


def _coverage_color(pct: float) -> str:
    """Map a coverage percentage to a shields.io badge color."""
    if pct >= 90:
        return "brightgreen"
    return "yellow" if pct >= 75 else "red"


def refresh_readme_badges() -> None:
    """Stamp the README coverage + test-count badges from CI's cumulative metrics."""
    metrics = _fetch_release_metrics()
    coverage_pct = float(metrics["coverage_pct"])
    union = int(metrics["test_union"])
    max_combo = int(metrics["test_max"])
    if max_combo and union > TEST_COUNT_UNION_RATIO_WARN * max_combo:
        print(
            f"\nWARNING: cumulative test count ({union}) exceeds "
            f"{TEST_COUNT_UNION_RATIO_WARN}x the largest single combo ({max_combo}). "
            "Node-id mismatches across combos can inflate the union — verify before publishing.\n",
            file=sys.stderr,
        )
    text = README.read_text()
    text = re.sub(
        r"badge/coverage-[\d.]+%25-[a-z]+",
        f"badge/coverage-{coverage_pct:.2f}%25-{_coverage_color(coverage_pct)}",
        text,
    )
    text = re.sub(r"badge/tests-\d+-blue", f"badge/tests-{union}-blue", text)
    README.write_text(text)


def stamp_readme_splash_url(version: str) -> None:
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


def stamp_splash(version: str) -> None:
    """Stamp the release version onto the committed splash webp (needs ImageMagick).

    Runs the version-overlay stage of ``create_splash_with_version.sh`` on the
    committed, version-independent base image. Fails loudly if ``magick`` is
    absent, since a maintainer-driven release must produce the real asset.
    """
    if shutil.which("magick") is None:
        fail_with_message("ImageMagick ('magick') is required to stamp the release splash but was not found")
    run_command(["sh", str(SPLASH_SCRIPT), f"v{version}"], cwd=REPO_ROOT)


def step_11_commit_release(version: str) -> None:
    """Refresh README badges, stamp the splash, then create the release commit."""
    print_step(11, f"refresh README badges + stamp splash + commit 'release: {version}'")
    refresh_readme_badges()
    stamp_readme_splash_url(version)
    stamp_splash(version)
    run_command(["git", "add", "pyproject.toml", "uv.lock", "CHANGELOG.md", "README.md", str(SPLASH_WEBP)])
    run_command(["git", "commit", "-m", f"release: {version}"])


def step_12_tag(version: str) -> None:
    """Create the version tag."""
    print_step(12, f"create tag v{version}")
    run_command(["git", "tag", f"v{version}"])


# ==================================================================================================
#  post-release steps (13-15)
# ==================================================================================================
def step_13_add_unreleased_section() -> None:
    """Add a fresh Unreleased section to the changelog."""
    print_step(13, "add fresh '## Unreleased' section to CHANGELOG.md")
    text = CHANGELOG.read_text()
    m = re.search(r"^## ", text, re.MULTILINE)
    if not m:
        fail_with_message("CHANGELOG.md has no version sections")
    insertion = "## Unreleased\n\n" + "\n".join(f"### {c}\n" for c in CATEGORIES) + "\n"
    text = text[: m.start()] + insertion + text[m.start() :]
    CHANGELOG.write_text(text)


def step_14_commit_next_cycle() -> None:
    """Commit the fresh Unreleased section."""
    print_step(14, "commit 'chore: begin next development cycle'")
    run_command(["git", "add", "CHANGELOG.md"])
    run_command(["git", "commit", "-m", "chore: begin next development cycle"])


def step_15_push(version: str) -> None:
    """Push main and the tag atomically."""
    print_step(15, f"push main + v{version} atomically")
    run_command(["git", "push", "--atomic", "origin", "main", f"refs/tags/v{version}"])


# ==================================================================================================
#  orchestration
# ==================================================================================================
def post_tag_recovery_hint(version: str, also_next_cycle: bool) -> None:
    """Print recovery instructions after a post-tag failure."""
    reset_count = 2 if also_next_cycle else 1
    print(
        f"\nERROR: a post-tag step failed.\n"
        f"Local state: release commit and tag v{version} created, not pushed.\n"
        f"To abort and retry:\n"
        f"  git tag -d v{version}\n"
        f"  git reset --hard HEAD~{reset_count}\n",
        file=sys.stderr,
    )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="X.Y.Z (no leading v)")
    args = parser.parse_args()
    version = args.version
    parse_semver(version)

    print(f"Releasing {PACKAGE_NAME} v{version}\n")

    print("Validation:")
    step_1_check_working_tree()
    step_2_check_in_sync()
    step_3_check_version_upgrade(version)
    step_4_check_tag_doesnt_exist(version)
    step_5_check_pypi_doesnt_have(version)
    step_6_check_classifiers_match()
    step_7_check_changelog_has_entries()

    print("\nRelease commit:")
    step_8_bump_version(version)
    step_9_lock()
    step_10_finalize_changelog(version)
    step_11_commit_release(version)
    step_12_tag(version)

    print("\nPost-release:")
    also_next_cycle = False
    try:
        step_13_add_unreleased_section()
        step_14_commit_next_cycle()
        also_next_cycle = True
        step_15_push(version)
    except subprocess.CalledProcessError:
        post_tag_recovery_hint(version, also_next_cycle)
        sys.exit(1)
    except SystemExit:
        post_tag_recovery_hint(version, also_next_cycle)
        raise


if __name__ == "__main__":
    main()
