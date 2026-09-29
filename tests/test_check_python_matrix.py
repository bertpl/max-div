"""Guards for the CI-matrix coverage check (scripts/check_python_matrix.py)."""

from pathlib import Path

from tests.helpers import load_script

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_python_matrix.py"


_mod = load_script("check_python_matrix", SCRIPT)


def test_reads_only_quoted_matrix_python_values(tmp_path):
    """The parser picks up quoted `python:` matrix legs and ignores `python_version:` / expressions."""
    # --- arrange ----------------------
    workflow = tmp_path / "wf.yml"
    workflow.write_text(
        "        include:\n"
        '          - { python: "3.11", resolution: highest }\n'
        '          - { python: "3.14t", resolution: highest }\n'
        "          python_version: ${{ matrix.python }}\n",
        encoding="utf-8",
    )

    # --- act --------------------------
    tested = _mod.read_tested_versions(workflow)

    # --- assert -----------------------
    assert tested == {"3.11", "3.14t"}


def test_uncovered_versions_flags_declared_gap_only():
    """A declared version absent from the matrix is uncovered; an extra matrix leg is not."""
    # --- act --------------------------
    missing = _mod.uncovered_versions(declared={"3.11", "3.15"}, tested={"3.11", "3.14t"})

    # --- assert -----------------------
    assert missing == {"3.15"}


def test_repo_matrix_covers_declared_versions():
    """The live repo satisfies the invariant: every declared version has a matrix leg."""
    # --- act --------------------------
    exit_code = _mod.main()

    # --- assert -----------------------
    assert exit_code == 0
