"""Build the CI test matrix from the Python version files and the `[tool.ci-test-matrix]` table in pyproject.toml.

The table holds a baseline job and a list of variants, for example:

    [tool.ci-test-matrix]
    baseline = { runner = "ubuntu-latest", resolution = "highest" }
    variants = [
        { pythons = "oldest+default", resolution = "lowest-direct" },
        { pythons = ["3.14"], is_free_threaded = true },
    ]

- The baseline runs on every Python in `.python-versions`. Its keys are the settings of a job, and
  every row of the matrix carries all of them.
- Each variant runs on a selection of Pythons and overrides some of the baseline's settings. A
  selection is an explicit list of Pythons, or names joined with `+`: `all`, `oldest`, `newest`, and
  `default`, the Python in `.python-version`.
- `is_free_threaded = true` runs the free-threaded build of each selected Python, such as `3.14t`.
- `env` holds the environment variables that each job of the variant sets for its test run.

The script holds no job or setting of its own: those live in the table. It uses only the standard
library, so it runs with `uv run --no-project`.

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

# These keys of a variant are not settings of a job.
NON_SETTING_VARIANT_KEYS = ("pythons", "is_free_threaded", "env")


# ==================================================================================================
#  CiTestMatrix
# ==================================================================================================
@dataclass(frozen=True)
class CiTestMatrix:
    """A CiTestMatrix holds the jobs of the CI test matrix, 1 row per job.

    The baseline's rows come first, then each variant's.
    """

    rows: tuple[Row, ...]

    def env_for(self, query: dict[str, str]) -> dict[str, str]:
        """Return the env of the rows whose settings match `query`, or an empty dict if no row matches.

        Raises:
            CiTestMatrixError: If `query` names a setting that the rows do not have, or if the
                matching rows set different env, so no single env answers the query.
        """
        envs: list[dict[str, str]] = []
        for row in self.rows:
            if row.matches(query) and row.env not in envs:
                envs.append(row.env)
        if len(envs) > 1:
            raise CiTestMatrixError(f"the rows matching {query} set different env: {envs}")
        if envs:
            return envs[0]
        else:
            return {}

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def from_repo(cls, root: Path) -> CiTestMatrix:
        """Build the matrix from the version files and the `[tool.ci-test-matrix]` table of the repo at `root`.

        Raises:
            CiTestMatrixError: If the table or a version file is invalid, or 2 rows are the same job.
        """
        versions = PythonVersions.from_repo(root)
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        table = pyproject.get("tool", {}).get("ci-test-matrix")
        if table is None:
            raise CiTestMatrixError("pyproject.toml has no [tool.ci-test-matrix] table")
        baseline = table.get("baseline", {})
        variants = [Variant.for_baseline(baseline)]
        variants += [Variant.from_table_entry(entry, baseline) for entry in table.get("variants", [])]
        rows: list[Row] = []
        for variant in variants:
            for row in variant.build_rows(versions, baseline):
                if row.to_matrix_entry() in [existing.to_matrix_entry() for existing in rows]:
                    raise CiTestMatrixError(f"2 variants produce the same job {row.to_matrix_entry()}")
                rows.append(row)
        return cls(tuple(rows))


# ==================================================================================================
#  Variant
# ==================================================================================================
@dataclass(frozen=True)
class Variant:
    """A variant is a set of jobs that override some of the baseline's settings, 1 job per selected Python."""

    pythons: str | list[str]
    overrides: dict[str, str]
    is_free_threaded: bool
    env: dict[str, str]

    def build_rows(self, versions: PythonVersions, baseline: dict[str, str]) -> list[Row]:
        """Return this variant's rows: the baseline's settings with the overrides, on each selected Python."""
        suffix = "t" if self.is_free_threaded else ""
        settings = {**baseline, **self.overrides}
        return [Row(python + suffix, settings, self.env) for python in versions.select(self.pythons)]

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _require_strings(values: dict, table_part: str) -> None:
        """Raise `CiTestMatrixError` if a value in `values` is not a string.

        The error message names `table_part`, the part of the table that holds `values`.
        """
        not_strings = sorted(key for key, value in values.items() if not isinstance(value, str))
        if not_strings:
            raise CiTestMatrixError(f"{table_part} sets {not_strings} to a value that is not a string")

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def for_baseline(cls, baseline: dict[str, str]) -> Variant:
        """Build the variant that runs on every Python and overrides nothing, so its rows are the baseline's.

        Raises:
            CiTestMatrixError: If the baseline sets a key that is not a setting of a job, or holds a
                value that is not a string.
        """
        non_settings = sorted(set(baseline) & {"python", *NON_SETTING_VARIANT_KEYS})
        if non_settings:
            raise CiTestMatrixError(f"the baseline sets {non_settings}, which are not settings of a job")
        cls._require_strings(baseline, "the baseline")
        return cls("all", {}, False, {})

    @classmethod
    def from_table_entry(cls, entry: dict, baseline: dict[str, str]) -> Variant:
        """Build a variant from 1 entry of the table's `variants` list.

        Raises:
            CiTestMatrixError: If the entry has no `pythons` key, overrides a setting that the
                baseline does not define, or holds a value that is not a string.
        """
        if "pythons" not in entry:
            raise CiTestMatrixError(f"the variant {entry} has no `pythons` selection")
        overrides = {key: value for key, value in entry.items() if key not in NON_SETTING_VARIANT_KEYS}
        unknown = sorted(set(overrides) - set(baseline))
        if unknown:
            raise CiTestMatrixError(f"the variant {entry} sets {unknown}, which the baseline does not define")
        cls._require_strings(overrides, f"the variant {entry}")
        env = entry.get("env", {})
        cls._require_strings(env, f"the env of the variant {entry}")
        return cls(entry["pythons"], overrides, entry.get("is_free_threaded", False), env)


# ==================================================================================================
#  Row
# ==================================================================================================
@dataclass(frozen=True)
class Row:
    """A row is one job of the matrix: its Python, its settings, and the env of its test run."""

    python: str
    settings: dict[str, str]
    env: dict[str, str]

    def to_matrix_entry(self) -> dict[str, str]:
        """Return the job as the workflow's matrix reads it: the Python and the settings, without the env."""
        return {"python": self.python, **self.settings}

    def matches(self, query: dict[str, str]) -> bool:
        """Return whether the job has every setting in `query` at the value that `query` gives.

        Raises:
            CiTestMatrixError: If `query` names a setting that the job does not have.
        """
        entry = self.to_matrix_entry()
        unknown = sorted(set(query) - set(entry))
        if unknown:
            raise CiTestMatrixError(f"the matrix rows have no setting {unknown}")
        return all(entry[key] == value for key, value in query.items())


# ==================================================================================================
#  PythonVersions
# ==================================================================================================
@dataclass(frozen=True)
class PythonVersions:
    """PythonVersions holds the supported Python versions, oldest first, and the default one."""

    supported: tuple[str, ...]
    default: str

    def select(self, selection: str | list[str]) -> tuple[str, ...]:
        """Return the Pythons in `selection`, oldest first, each once.

        Args:
            selection: An explicit list of supported Pythons, or names joined with `+`: `all`,
                `oldest`, `newest` and `default`.

        Raises:
            CiTestMatrixError: If the selection names an unknown name or an unsupported Python.
        """
        if isinstance(selection, list):
            selected = set(selection)
            unsupported = sorted(selected - set(self.supported))
            if unsupported:
                raise CiTestMatrixError(f"{unsupported} are not in .python-versions {list(self.supported)}")
        else:
            selected = set()
            for name in selection.split("+"):
                selected.update(self._select_by_name(name))
        return tuple(python for python in self.supported if python in selected)

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    def _select_by_name(self, name: str) -> tuple[str, ...]:
        """Return the Pythons of 1 name in a `+`-joined selection."""
        if name == "all":
            return self.supported
        elif name == "oldest":
            return (self.supported[0],)
        elif name == "newest":
            return (self.supported[-1],)
        elif name == "default":
            return (self.default,)
        else:
            raise CiTestMatrixError(f"unknown Python selection {name!r}; use all, oldest, newest, default or a list")

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def from_repo(cls, root: Path) -> PythonVersions:
        """Read the supported Pythons from `.python-versions` and the default one from `.python-version`.

        Raises:
            CiTestMatrixError: If the default Python is not a supported one.
        """
        lines = (root / ".python-versions").read_text(encoding="utf-8").splitlines()
        supported = sorted(
            (line.strip() for line in lines if line.strip()),
            key=lambda python: tuple(int(part) for part in python.split(".")),
        )
        default = (root / ".python-version").read_text(encoding="utf-8").strip()
        if default not in supported:
            raise CiTestMatrixError(f"the default Python {default} is not in .python-versions {supported}")
        return cls(tuple(supported), default)


# ==================================================================================================
#  CiTestMatrixError
# ==================================================================================================
class CiTestMatrixError(ValueError):
    """The `[tool.ci-test-matrix]` table, a version file that it relies on, or a command-line argument is invalid."""


# ==================================================================================================
#  Command line
# ==================================================================================================
def main(argv: list[str] | None = None) -> int:
    """Run the subcommand in `argv`.

    Returns:
        1 if the table, a version file or a command-line setting is invalid, 0 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root to read the files from; defaults to this checkout",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("matrix", help="print the rows as a JSON list, 1 object per job")
    env_parser = subparsers.add_parser("env", help="print the env of the rows that match the given KEY=VALUE query")
    env_parser.add_argument("query", nargs="+", metavar="KEY=VALUE")
    subparsers.add_parser("check", help="validate the table against the version files")
    args = parser.parse_args(argv)

    try:
        matrix = CiTestMatrix.from_repo(args.root)
        if args.command == "matrix":
            print(json.dumps([row.to_matrix_entry() for row in matrix.rows]))
        elif args.command == "env":
            env = matrix.env_for(_parse_query(args.query))
            print(" ".join(f"{key}={value}" for key, value in env.items()))
        else:
            print(f"CI test matrix OK: {len(matrix.rows)} jobs")
    except CiTestMatrixError as error:
        print(f"ERROR  {error}", file=sys.stderr)
        return 1
    else:
        return 0


def _parse_query(items: list[str]) -> dict[str, str]:
    """Turn `KEY=VALUE` command-line items into the query dict of `CiTestMatrix.env_for`.

    Raises:
        CiTestMatrixError: If an item has no `=`.
    """
    query: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise CiTestMatrixError(f"{item!r} is not KEY=VALUE")
        query[key] = value
    return query


if __name__ == "__main__":
    sys.exit(main())
