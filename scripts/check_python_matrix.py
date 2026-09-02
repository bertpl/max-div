"""Check that the CI test matrix covers every Python in `.python-versions`.

`.python-versions` is the declared set of supported Python minors, and `scripts/release.py` ties it
to the trove classifiers. Nothing ties it to what CI runs.

The test matrix in `.github/workflows/_unit_tests.yml` is separate and hand-curated (each Python
paired with resolution and jit axes, plus a free-threaded `3.14t` leg), so a version can sit in
`.python-versions` and the classifiers, pass the release check, and reach PyPI with no test leg
running on it.

This check fails when a declared version has no matrix leg. It is one-directional: extra matrix legs
(notably `3.14t`, deliberately absent from `.python-versions`) are fine; only an uncovered declared
version is an error.

Usage:

    python scripts/check_python_matrix.py    # exits non-zero if a declared version is untested
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_VERSIONS_FILE = REPO_ROOT / ".python-versions"
UNIT_TESTS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "_unit_tests.yml"

# Matrix legs quote their Python as `python: "3.12"`. The different key `python_version:` and the
# `${{ ... }}` references carry no quoted literal, so this matches the matrix versions and nothing
# else.
_MATRIX_PYTHON = re.compile(r'\bpython:\s*"([^"]+)"')


def read_declared_versions(path: Path = PYTHON_VERSIONS_FILE) -> set[str]:
    """Return the Python minors declared in `.python-versions` (one per non-empty line)."""
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_tested_versions(path: Path = UNIT_TESTS_WORKFLOW) -> set[str]:
    """Return the Python value of every leg in the workflow's test matrix."""
    return set(_MATRIX_PYTHON.findall(path.read_text(encoding="utf-8")))


def uncovered_versions(declared: set[str], tested: set[str]) -> set[str]:
    """Return declared versions with no matching matrix leg; extra tested legs are allowed."""
    return declared - tested


def main() -> int:
    """Report any declared Python version the CI matrix does not test; return 1 if any."""
    declared = read_declared_versions()
    tested = read_tested_versions()
    missing = uncovered_versions(declared, tested)
    if missing:
        print(
            f".python-versions declares {sorted(missing)} with no matching leg in the "
            f"{UNIT_TESTS_WORKFLOW.relative_to(REPO_ROOT)} test matrix (matrix tests {sorted(tested)}).",
            file=sys.stderr,
        )
        return 1
    print(f"CI matrix covers all {len(declared)} declared Python versions: {sorted(declared)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
