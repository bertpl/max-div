"""These tests cover the rows that scripts/ci_test_matrix.py builds, its checks, and max-div's table."""

import json
from pathlib import Path

import pytest

from scripts.tests.helpers import load_script

_mod = load_script("ci_test_matrix")

_SUPPORTED = ("3.12", "3.13", "3.14", "3.15")

_TABLE = """
[tool.ci-test-matrix]
baseline = { resolution = "highest", jit = "on" }
variants = [
    { pythons = "default", resolution = "locked", env = { GOLDEN = "1" } },
    { pythons = "oldest+default", jit = "off" },
    { pythons = ["3.14"], is_free_threaded = true },
]
"""


def _write_repo(root: Path, pyproject: str, default: str = "3.14") -> Path:
    """Write the 2 version files and a pyproject.toml holding `pyproject` under `root`; return `root`."""
    (root / ".python-versions").write_text("\n".join(_SUPPORTED) + "\n", encoding="utf-8")
    (root / ".python-version").write_text(f"{default}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return root


def test_the_rows_are_the_baseline_on_every_python_then_each_variant(tmp_path):
    """Each variant's rows carry the baseline's settings with its overrides, on its selected Pythons."""
    # --- arrange ----------------------
    root = _write_repo(tmp_path, _TABLE)

    # --- act --------------------------
    matrix = _mod.CiTestMatrix.from_repo(root)

    # --- assert -----------------------
    assert [row.to_matrix_entry() for row in matrix.rows] == [
        {"python": "3.12", "resolution": "highest", "jit": "on"},
        {"python": "3.13", "resolution": "highest", "jit": "on"},
        {"python": "3.14", "resolution": "highest", "jit": "on"},
        {"python": "3.15", "resolution": "highest", "jit": "on"},
        {"python": "3.14", "resolution": "locked", "jit": "on"},
        {"python": "3.12", "resolution": "highest", "jit": "off"},
        {"python": "3.14", "resolution": "highest", "jit": "off"},
        {"python": "3.14t", "resolution": "highest", "jit": "on"},
    ]


@pytest.mark.parametrize(
    "selection, expected",
    [
        pytest.param("all", _SUPPORTED, id="all"),
        pytest.param("oldest", ("3.12",), id="oldest"),
        pytest.param("newest", ("3.15",), id="newest"),
        pytest.param("default", ("3.14",), id="default"),
        pytest.param("default+oldest", ("3.12", "3.14"), id="joined_names_oldest_first"),
        pytest.param("newest+all", _SUPPORTED, id="overlapping_names_each_once"),
        pytest.param(["3.15", "3.13"], ("3.13", "3.15"), id="explicit_list_oldest_first"),
    ],
)
def test_a_selection_picks_its_pythons_oldest_first_each_once(selection, expected):
    """A selection returns its Pythons oldest first, each once."""
    # --- arrange ----------------------
    versions = _mod.PythonVersions(_SUPPORTED, "3.14")

    # --- act / assert -----------------
    assert versions.select(selection) == expected


def test_the_supported_pythons_are_sorted_by_version_not_as_text(tmp_path):
    """The supported Pythons are sorted by version, so 3.9 comes before 3.10."""
    # --- arrange ----------------------
    (tmp_path / ".python-versions").write_text("3.10\n3.9\n", encoding="utf-8")
    (tmp_path / ".python-version").write_text("3.10\n", encoding="utf-8")

    # --- act --------------------------
    versions = _mod.PythonVersions.from_repo(tmp_path)

    # --- assert -----------------------
    assert versions.supported == ("3.9", "3.10")


@pytest.mark.parametrize(
    "query, expected",
    [
        pytest.param({"python": "3.14", "resolution": "locked"}, {"GOLDEN": "1"}, id="row_with_env"),
        pytest.param({"python": "3.14", "resolution": "highest"}, {}, id="rows_without_env"),
        pytest.param({"python": "3.13", "resolution": "locked"}, {}, id="no_matching_row"),
    ],
)
def test_env_for_returns_the_env_of_the_matching_rows(tmp_path, query, expected):
    """env_for returns the env of the matching rows, or an empty dict if no row matches."""
    # --- arrange ----------------------
    matrix = _mod.CiTestMatrix.from_repo(_write_repo(tmp_path, _TABLE))

    # --- act / assert -----------------
    assert matrix.env_for(query) == expected


@pytest.mark.parametrize(
    "query, message",
    [
        pytest.param({"python": "3.14"}, "set different env", id="rows_that_disagree"),
        pytest.param({"os": "linux"}, "no setting", id="unknown_setting"),
    ],
)
def test_env_for_raises_when_the_query_does_not_decide_the_env(tmp_path, query, message):
    """env_for raises when the matching rows disagree or the query names an unknown setting."""
    # --- arrange ----------------------
    matrix = _mod.CiTestMatrix.from_repo(_write_repo(tmp_path, _TABLE))

    # --- act / assert -----------------
    with pytest.raises(_mod.CiTestMatrixError, match=message):
        matrix.env_for(query)


@pytest.mark.parametrize(
    "pyproject, default, message",
    [
        pytest.param("[tool.other]\n", "3.14", r"no \[tool.ci-test-matrix\] table", id="no_table"),
        pytest.param(
            '[tool.ci-test-matrix]\nbaseline = { python = "3.14" }\n', "3.14", "not settings", id="reserved_key"
        ),
        pytest.param("[tool.ci-test-matrix]\nbaseline = { jit = true }\n", "3.14", "not a string", id="non_string"),
        pytest.param(
            '[tool.ci-test-matrix]\nbaseline = { jit = "on" }\nvariants = [{ jit = "off" }]\n',
            "3.14",
            "no `pythons` selection",
            id="variant_without_pythons",
        ),
        pytest.param(
            '[tool.ci-test-matrix]\nbaseline = { jit = "on" }\nvariants = [{ pythons = "all", os = "mac" }]\n',
            "3.14",
            "does not define",
            id="variant_setting_not_in_baseline",
        ),
        pytest.param(
            '[tool.ci-test-matrix]\nbaseline = { jit = "on" }\nvariants = [{ pythons = "all", env = { X = 1 } }]\n',
            "3.14",
            "not a string",
            id="non_string_env",
        ),
        pytest.param(
            '[tool.ci-test-matrix]\nbaseline = { jit = "on" }\nvariants = [{ pythons = "middle", jit = "off" }]\n',
            "3.14",
            "unknown Python selection",
            id="unknown_selection_name",
        ),
        pytest.param(
            '[tool.ci-test-matrix]\nbaseline = { jit = "on" }\nvariants = [{ pythons = ["3.11"], jit = "off" }]\n',
            "3.14",
            r"\['3.11'\] are not in .python-versions",
            id="unsupported_python",
        ),
        pytest.param(
            '[tool.ci-test-matrix]\nbaseline = { jit = "on" }\nvariants = [{ pythons = "default" }]\n',
            "3.14",
            "the same job",
            id="duplicate_row",
        ),
        pytest.param(_TABLE, "3.11", "default Python 3.11 is not in", id="unsupported_default"),
    ],
)
def test_an_invalid_table_or_version_file_raises(tmp_path, pyproject, default, message):
    """An invalid table or version file raises CiTestMatrixError."""
    # --- arrange ----------------------
    root = _write_repo(tmp_path, pyproject, default=default)

    # --- act / assert -----------------
    with pytest.raises(_mod.CiTestMatrixError, match=message):
        _mod.CiTestMatrix.from_repo(root)


def test_the_matrix_command_prints_the_rows_as_json(tmp_path, capsys):
    """The matrix command prints the rows as a JSON list and exits 0."""
    # --- arrange ----------------------
    root = _write_repo(tmp_path, _TABLE)

    # --- act --------------------------
    exit_code = _mod.main(["--root", str(root), "matrix"])

    # --- assert -----------------------
    rows = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert rows == [row.to_matrix_entry() for row in _mod.CiTestMatrix.from_repo(root).rows]


@pytest.mark.parametrize(
    "args, expected_stdout",
    [
        pytest.param(["env", "python=3.14", "resolution=locked"], "GOLDEN=1\n", id="env_with_a_value"),
        pytest.param(["env", "python=3.14", "resolution=highest"], "\n", id="env_without_a_value"),
        pytest.param(["check"], "CI test matrix OK: 8 jobs\n", id="check"),
    ],
)
def test_the_env_and_check_commands_print_their_result(tmp_path, capsys, args, expected_stdout):
    """The env and check commands print their result and exit 0."""
    # --- arrange ----------------------
    root = _write_repo(tmp_path, _TABLE)

    # --- act --------------------------
    exit_code = _mod.main(["--root", str(root), *args])

    # --- assert -----------------------
    assert exit_code == 0
    assert capsys.readouterr().out == expected_stdout


@pytest.mark.parametrize(
    "pyproject, args",
    [
        pytest.param("[tool.other]\n", ["check"], id="invalid_table"),
        pytest.param(_TABLE, ["env", "python"], id="setting_without_a_value"),
    ],
)
def test_a_command_reports_an_error_and_exits_1(tmp_path, capsys, pyproject, args):
    """A command that hits a CiTestMatrixError prints ERROR to stderr and exits 1."""
    # --- arrange ----------------------
    root = _write_repo(tmp_path, pyproject)

    # --- act --------------------------
    exit_code = _mod.main(["--root", str(root), *args])

    # --- assert -----------------------
    assert exit_code == 1
    assert capsys.readouterr().err.startswith("ERROR  ")


def test_the_live_matrix_collects_coverage_on_exactly_its_jit_off_jobs():
    """Exactly the jit-off jobs collect coverage.

    Coverage traces the lines inside numba-compiled functions only with JIT compilation off.
    """
    # --- act --------------------------
    matrix = _mod.CiTestMatrix.from_repo(_mod.REPO_ROOT)

    # --- assert -----------------------
    coverage_rows = [row for row in matrix.rows if row.settings["coverage"] == "true"]
    jit_off_rows = [row for row in matrix.rows if row.settings["jit"] == "off"]
    assert coverage_rows
    assert coverage_rows == jit_off_rows


def test_the_live_matrix_runs_the_jit_golden_master_on_1_locked_jit_on_job_of_the_default_python():
    """Only the locked jit-on job of the default Python runs the JIT golden master.

    Installing from uv.lock on the default Python gives the Python and numba versions that the JIT golden master
    recorded.
    """
    # --- arrange ----------------------
    default = _mod.PythonVersions.from_repo(_mod.REPO_ROOT).default

    # --- act --------------------------
    matrix = _mod.CiTestMatrix.from_repo(_mod.REPO_ROOT)

    # --- assert -----------------------
    golden_master_rows = [row for row in matrix.rows if row.env.get("MAX_DIV_JIT_GOLDEN_MASTER") == "1"]
    assert [(row.python, row.settings["resolution"], row.settings["jit"]) for row in golden_master_rows] == [
        (default, "locked", "on")
    ]
