"""Build the CI test matrix from the Python version files and the `[tool.ci-test-matrix]` table in pyproject.toml.

The table holds a baseline job and a list of variants, for example:

    [tool.ci-test-matrix]
    baseline = { runner = "ubuntu-latest", resolution = "highest" }
    variants = [
        { pythons = "oldest+default", resolution = "lowest-direct" },
        { pythons = ["3.14"], free_threaded = true },
    ]

- The baseline runs on every Python in `.python-versions`. Its keys are the settings of a job, and
  every row of the matrix carries all of them.
- Each variant runs on a selection of Pythons and overrides some of the baseline's settings. A
  selection is an explicit list of Pythons, or names joined with `+`: `all`, `oldest`, `newest`, and
  `default`, the Python in `.python-version`.
- `free_threaded = true` runs the free-threaded build of each selected Python, such as `3.14t`.
- `env` holds the environment variables that a test run with the variant's settings sets.

The script holds no project-specific rule, and uses only the standard library, so it runs with
`uv run --no-project`.

Usage:
    python scripts/ci_test_matrix.py matrix                               # the rows as JSON, 1 object per job
    python scripts/ci_test_matrix.py env python=3.14 resolution=locked    # the env of the matching rows
    python scripts/ci_test_matrix.py check                                # validate the table only
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys of a variant that are not settings of a job.
RESERVED_VARIANT_KEYS = ("pythons", "free_threaded", "env")


# ==================================================================================================
#  CiTestMatrix
# ==================================================================================================
@dataclass(frozen=True)
class CiTestMatrix:
    """The jobs of the CI test matrix, 1 row per job: the baseline's rows first, then each variant's."""

    rows: tuple[Row, ...]

    def env_for(self, query: dict[str, str]) -> dict[str, str]:
        """Return the env of the rows whose settings match `query`, or an empty dict if no row matches.

        Raises:
            MatrixConfigError: If `query` names a setting that the rows do not have, or if the
                matching rows set different env, so that the query does not decide the env.
        """
        unknown = set(query) - set(self.rows[0].to_matrix_entry())
        if unknown:
            raise MatrixConfigError(f"the matrix rows have no setting {sorted(unknown)}")
        envs: list[dict[str, str]] = []
        for row in self.rows:
            if row.matches(query) and row.env not in envs:
                envs.append(row.env)
        if len(envs) > 1:
            raise MatrixConfigError(f"the rows matching {query} set different env: {envs}")
        if envs:
            return envs[0]
        else:
            return {}

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def from_config(cls, root: Path) -> CiTestMatrix:
        """Build the matrix from the version files and the `[tool.ci-test-matrix]` table under `root`.

        Raises:
            MatrixConfigError: If the table or a version file is invalid, or 2 rows are the same job.
        """
        versions = PythonVersions.from_files(root)
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        table = pyproject.get("tool", {}).get("ci-test-matrix")
        if table is None:
            raise MatrixConfigError("pyproject.toml has no [tool.ci-test-matrix] table")
        baseline = table.get("baseline", {})
        reserved = sorted(set(baseline) & {"python", *RESERVED_VARIANT_KEYS})
        if reserved:
            raise MatrixConfigError(f"the baseline sets {reserved}, which are not settings of a job")
        _require_strings(baseline, "the baseline")

        # The baseline's own rows are those of a variant that overrides nothing on every Python.
        variants = [Variant("all", {}, False, {})]
        variants += [Variant.from_table(variant, baseline) for variant in table.get("variants", [])]
        rows: list[Row] = []
        for variant in variants:
            for row in variant.rows(versions, baseline):
                if row.to_matrix_entry() in [existing.to_matrix_entry() for existing in rows]:
                    raise MatrixConfigError(f"2 variants produce the same job {row.to_matrix_entry()}")
                rows.append(row)
        return cls(tuple(rows))


# ==================================================================================================
#  Variant
# ==================================================================================================
@dataclass(frozen=True)
class Variant:
    """A set of jobs that override some of the baseline's settings, 1 job per selected Python."""

    pythons: str | list[str]
    overrides: dict[str, str]
    is_free_threaded: bool
    env: dict[str, str]

    def rows(self, versions: PythonVersions, baseline: dict[str, str]) -> list[Row]:
        """Return this variant's rows: the baseline's settings with the overrides, on each selected Python."""
        suffix = "t" if self.is_free_threaded else ""
        settings = {**baseline, **self.overrides}
        return [Row(python + suffix, settings, self.env) for python in versions.select(self.pythons)]

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def from_table(cls, table: dict, baseline: dict[str, str]) -> Variant:
        """Build a variant from 1 entry of the table's `variants` list.

        Raises:
            MatrixConfigError: If the entry selects no Pythons, overrides a setting that the
                baseline does not define, or holds a value that is not a string.
        """
        if "pythons" not in table:
            raise MatrixConfigError(f"the variant {table} has no `pythons` selection")
        overrides = {key: value for key, value in table.items() if key not in RESERVED_VARIANT_KEYS}
        unknown = sorted(set(overrides) - set(baseline))
        if unknown:
            raise MatrixConfigError(f"the variant {table} sets {unknown}, which the baseline does not define")
        _require_strings(overrides, f"the variant {table}")
        env = table.get("env", {})
        _require_strings(env, f"the env of the variant {table}")
        return cls(table["pythons"], overrides, table.get("free_threaded", False), env)


# ==================================================================================================
#  Row
# ==================================================================================================
@dataclass(frozen=True)
class Row:
    """One job of the matrix: its Python, its settings, and the env that its test run sets."""

    python: str
    settings: dict[str, str]
    env: dict[str, str]

    def to_matrix_entry(self) -> dict[str, str]:
        """Return the job as the workflow's matrix reads it: the Python and the settings, without the env."""
        return {"python": self.python, **self.settings}

    def matches(self, query: dict[str, str]) -> bool:
        """Return whether the job has every setting in `query` at the value that `query` gives."""
        entry = self.to_matrix_entry()
        return all(entry[key] == value for key, value in query.items())


# ==================================================================================================
#  PythonVersions
# ==================================================================================================
@dataclass(frozen=True)
class PythonVersions:
    """The supported Python versions, oldest first, and the default one."""

    supported: tuple[str, ...]
    default: str

    def select(self, selection: str | list[str]) -> tuple[str, ...]:
        """Return the Pythons that a variant's `pythons` value selects, oldest first, each once.

        Args:
            selection: An explicit list of supported Pythons, or names joined with `+`: `all`,
                `oldest`, `newest` and `default`.

        Raises:
            MatrixConfigError: If the selection names an unknown name or an unsupported Python.
        """
        if isinstance(selection, list):
            selected = set(selection)
            unsupported = sorted(selected - set(self.supported))
            if unsupported:
                raise MatrixConfigError(f"{unsupported} are not in .python-versions {list(self.supported)}")
        else:
            selected = set()
            for name in selection.split("+"):
                selected.update(self._select_by_name(name))
        return tuple(python for python in self.supported if python in selected)

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    def _select_by_name(self, name: str) -> tuple[str, ...]:
        """Return the Pythons that 1 name of a `+`-joined selection stands for."""
        if name == "all":
            return self.supported
        elif name == "oldest":
            return (self.supported[0],)
        elif name == "newest":
            return (self.supported[-1],)
        elif name == "default":
            return (self.default,)
        else:
            raise MatrixConfigError(f"unknown Python selection {name!r}; use all, oldest, newest, default or a list")

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def from_files(cls, root: Path) -> PythonVersions:
        """Read the supported Pythons from `.python-versions` and the default one from `.python-version`.

        Raises:
            MatrixConfigError: If the default Python is not a supported one.
        """
        lines = (root / ".python-versions").read_text(encoding="utf-8").splitlines()
        supported = sorted(
            (line.strip() for line in lines if line.strip()),
            key=lambda python: tuple(int(part) for part in python.split(".")),
        )
        default = (root / ".python-version").read_text(encoding="utf-8").strip()
        if default not in supported:
            raise MatrixConfigError(f"the default Python {default} is not in .python-versions {supported}")
        return cls(tuple(supported), default)


# ==================================================================================================
#  MatrixConfigError
# ==================================================================================================
class MatrixConfigError(ValueError):
    """The `[tool.ci-test-matrix]` table, or a version file that it relies on, is invalid."""


# ==================================================================================================
#  Helpers
# ==================================================================================================
def _require_strings(values: dict, where: str) -> None:
    """Raise `MatrixConfigError` if a value in `values` is not a string, naming `where` it sits."""
    not_strings = sorted(key for key, value in values.items() if not isinstance(value, str))
    if not_strings:
        raise MatrixConfigError(f"{where} sets {not_strings} to a value that is not a string")


def _parse_settings(items: list[str]) -> dict[str, str]:
    """Turn `KEY=VALUE` command-line items into a dict.

    Raises:
        MatrixConfigError: If an item has no `=`.
    """
    settings: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise MatrixConfigError(f"{item!r} is not KEY=VALUE")
        settings[key] = value
    return settings


# ==================================================================================================
#  Command line
# ==================================================================================================
def main(argv: list[str] | None = None) -> int:
    """Run the subcommand in `argv`; return 1 if the table is invalid, 0 otherwise."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root to read the files from; defaults to this checkout",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("matrix", help="print the rows as a JSON list, 1 object per job")
    env_parser = subparsers.add_parser("env", help="print the env of the rows that match the given settings")
    env_parser.add_argument("settings", nargs="+", metavar="KEY=VALUE")
    subparsers.add_parser("check", help="validate the table against the version files")
    args = parser.parse_args(argv)

    try:
        matrix = CiTestMatrix.from_config(args.root)
        if args.command == "matrix":
            print(json.dumps([row.to_matrix_entry() for row in matrix.rows]))
        elif args.command == "env":
            env = matrix.env_for(_parse_settings(args.settings))
            print(" ".join(f"{key}={value}" for key, value in env.items()))
        else:
            print(f"CI test matrix OK: {len(matrix.rows)} jobs")
    except MatrixConfigError as error:
        print(f"ERROR  {error}", file=sys.stderr)
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
