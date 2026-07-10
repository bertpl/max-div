"""Compute release metrics from the combined CI coverage + node-id data.

Run by the coverage job after ``coverage combine``, with the combined
``.coverage`` and every combo's ``test-ids-*.txt`` present in the working
directory. Writes a metrics JSON (path from ``argv[1]``) with the numbers the
release stamps into the README badges: ``coverage_pct`` (combined matrix total),
``test_union`` (distinct test node-ids across all combos), and ``test_max`` (the
largest single-combo count).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _coverage_total() -> float:
    """Return the combined coverage percentage from the current ``.coverage``."""
    # --fail-under=0: just read the number; the gate is enforced by the combine job's report step
    out = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--format=total", "--fail-under=0"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(out.strip())


def _test_counts() -> tuple[int, int]:
    """Return ``(union, max)`` test-node-id counts across every combo's dump.

    Each combo dumps its own collected node-ids (``test-ids-*.txt``); the union
    is the true distinct-test count across the matrix, robust to any combo that
    collects a divergent set.
    """
    id_files = sorted(Path().glob("test-ids-*.txt"))
    if not id_files:
        sys.exit("no test-ids-*.txt files found to union")
    per_combo = {f.name: {ln.strip() for ln in f.read_text().splitlines() if "::" in ln} for f in id_files}
    for name, ids in per_combo.items():
        print(f"  {name}: {len(ids)}")
    union: set[str] = set().union(*per_combo.values())
    return len(union), max(len(ids) for ids in per_combo.values())


def main() -> None:
    """Compute metrics and write them to the JSON path in ``argv[1]``."""
    out_path = Path(sys.argv[1])
    test_union, test_max = _test_counts()
    metrics = {"coverage_pct": _coverage_total(), "test_union": test_union, "test_max": test_max}
    print(json.dumps(metrics, indent=2))
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
