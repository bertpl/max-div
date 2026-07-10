"""Extract a single version's section from CHANGELOG.md and print to stdout.

Used by release_tag.yml to populate the GitHub Release notes from the
just-released section in CHANGELOG.md.  The version is taken from the
git tag of the current commit (e.g. v0.1.0 -> 0.1.0).  If no tag is
present, exits non-zero with a clear error.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def current_tag_version() -> str:
    """Read the version string from the git tag on HEAD."""
    result = subprocess.run(
        ["git", "describe", "--exact-match", "--tags", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit("error: HEAD is not at a tag")
    tag = result.stdout.strip()
    if not tag.startswith("v"):
        sys.exit(f"error: tag {tag!r} does not start with 'v'")
    return tag[1:]


def extract_section(text: str, version: str) -> str:
    """Extract the changelog body for a specific version."""
    pattern = re.compile(
        rf"^## {re.escape(version)}\b.*?$(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        sys.exit(f"error: no '## {version}' section in CHANGELOG.md")
    return match.group(1).strip() + "\n"


def main() -> None:
    """Entry point."""
    version = current_tag_version()
    sys.stdout.write(extract_section(CHANGELOG.read_text(), version))


if __name__ == "__main__":
    main()
