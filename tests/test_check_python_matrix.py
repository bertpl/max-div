"""Guard the CI test matrix: every declared Python has a job, and the `resolution: locked` job runs DEFAULT_PY."""

import re

from tests.helpers import load_script

_mod = load_script("check_python_matrix")


def test_reads_only_quoted_matrix_python_values(tmp_path):
    """The parser picks up quoted `python:` matrix entries and ignores `python_version:` / expressions."""
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
    """A declared version absent from the matrix is uncovered; an extra matrix job is not."""
    # --- act --------------------------
    missing = _mod.uncovered_versions(declared={"3.11", "3.15"}, tested={"3.11", "3.14t"})

    # --- assert -----------------------
    assert missing == {"3.15"}


def test_repo_matrix_covers_declared_versions():
    """The live repo satisfies the invariant: every declared version has a matrix job."""
    # --- act --------------------------
    exit_code = _mod.main()

    # --- assert -----------------------
    assert exit_code == 0


def test_locked_matrix_job_runs_the_makefile_default_python():
    """The `resolution: locked` matrix job runs DEFAULT_PY, so `make test` runs the JIT golden master there."""
    # --- arrange ----------------------
    makefile = (_mod.REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = _mod.UNIT_TESTS_WORKFLOW.read_text(encoding="utf-8")

    # --- act --------------------------
    default_py = re.search(r"^DEFAULT_PY := (\S+)$", makefile, re.MULTILINE).group(1)
    locked_pythons = re.findall(r'python:\s*"([^"]+)",\s*resolution:\s*locked\b', workflow)

    # --- assert -----------------------
    assert locked_pythons == [default_py]
