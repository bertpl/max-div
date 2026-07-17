"""Tier-1 report emission: turn the full-scenario results into docs tables.

Run with: ``uv run --group benchmarks python -m benchmarks.tier1.report``.
Reads the result files written by ``benchmarks.tier1.full`` and emits markdown tables into
``docs/benchmarks/comparison/results/``. Only these curated tables are committed; the raw
results stay untracked but reproducible.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.common.records import RunRecord, load_records

RECORDS_DIR = Path("reports/benchmarks/tier1")
RESULTS_DIR = Path("docs/benchmarks/comparison/results")

# Ladder rungs quoted in the gap table (seconds; must be actual ladder rungs).
GAP_BUDGETS_SEC = (0.016, 0.128, 1.024, 16.384)


def maxdiv_gap_pct(records: list[RunRecord], metric_name: str, budget_sec: float, optimum: float) -> float:
    """Mean gap (%) of max-div to a proven optimum at one ladder rung (positive = below optimum)."""
    tag = f"time:{budget_sec}s"
    values = [r.quality[metric_name] for r in records if r.budget == tag]
    return (optimum - float(np.mean(values))) / optimum * 100.0


def build_maxmin_gap_table(exact_rows: list[dict], records: list[RunRecord]) -> str:
    """Markdown table: per problem, the proven optimum, its proof cost, and max-div's gaps."""
    by_key: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for r in records:
        by_key[(r.problem, r.size)].append(r)

    budget_headers = " | ".join(f"gap @{b:g}s" for b in GAP_BUDGETS_SEC)
    # 6 fixed columns (problem, n, k, m, optimum, proof time) plus one per budget.
    lines = [
        f"| problem | n | k | m | optimum | proof time | {budget_headers} |",
        "|---" * (6 + len(GAP_BUDGETS_SEC)) + "|",
    ]
    for row in exact_rows:
        recs = by_key[(row["problem"], row["size"])]
        gaps = " | ".join(f"{maxdiv_gap_pct(recs, 'MIN_SEPARATION', b, row['optimum']):.1f}%" for b in GAP_BUDGETS_SEC)
        lines.append(
            f"| {row['problem']} | {row['n']} | {row['k']} | {row['m']} | {row['optimum']:.4f} "
            f"| {row['measured_sec']:.2f} s | {gaps} |"
        )
    return "\n".join(lines) + "\n"


def build_scaling_table(rows: list[dict]) -> str:
    """Markdown table: time-to-proof per backend across the n-ladder ('timeout' where unproven)."""
    backends = list(dict.fromkeys(row["backend"] for row in rows))
    by_backend: dict[str, dict[int, dict]] = defaultdict(dict)
    ns: list[int] = sorted({row["n"] for row in rows})
    for row in rows:
        by_backend[row["backend"]][row["n"]] = row

    lines = [
        "| n | k | " + " | ".join(backends) + " |",
        "|---" * (2 + len(backends)) + "|",
    ]
    for n in ns:
        k = next(row["k"] for row in rows if row["n"] == n)
        cells = []
        for backend in backends:
            row = by_backend[backend].get(n)
            if row is None:
                cells.append("—")  # ladder stopped earlier for this backend
            elif row["proven"]:
                cells.append(f"{row['measured_sec']:.1f} s")
            else:
                cells.append("**timeout**")
        lines.append(f"| {n} | {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_incumbent_table(panel_rows: list[dict], records: list[RunRecord]) -> str:
    """Markdown table: CP-SAT's incumbent at its cap vs. max-div's ladder (uncertified)."""
    by_key: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for r in records:
        by_key[(r.problem, r.size)].append(r)

    lines = [
        "| problem | n | k | m | CP-SAT cap | CP-SAT incumbent | bound gap | max-div @~1 s (best seed) |",
        "|---" * 8 + "|",
    ]
    for row in panel_rows:
        recs = [r for r in by_key[(row["problem"], row["size"])] if r.budget == "time:1.024s"]
        best_1s = max(r.quality["GEOMEAN_SEPARATION"] for r in recs)
        bound_gap = (row["objective_bound"] - row["objective_value"]) / row["objective_value"] * 100.0
        lines.append(
            f"| {row['problem']} | {row['n']} | {row['k']} | {row['m']} | {row['cap_sec']:.0f} s "
            f"| {row['objective_value']:.4f} | {bound_gap:.0f}% | {best_1s:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Emit all tier-1 docs tables from the recorded results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    exact_rows = json.loads((RECORDS_DIR / "maxmin_exact.json").read_text())
    maxmin_records = load_records(RECORDS_DIR / "maxmin_records.jsonl")
    (RESULTS_DIR / "tier1_maxmin_gap.md").write_text(build_maxmin_gap_table(exact_rows, maxmin_records))

    scaling_rows = json.loads((RECORDS_DIR / "scaling.json").read_text())
    (RESULTS_DIR / "tier1_scaling.md").write_text(build_scaling_table(scaling_rows))

    panel_rows = json.loads((RECORDS_DIR / "incumbent.json").read_text())
    incumbent_records = load_records(RECORDS_DIR / "incumbent_records.jsonl")
    (RESULTS_DIR / "tier1_incumbent_geomean.md").write_text(build_incumbent_table(panel_rows, incumbent_records))

    print(f"tier-1 report emitted into {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
