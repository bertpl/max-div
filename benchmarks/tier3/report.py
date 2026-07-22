"""Tier-3 report emission: turn the MDPLIB records into docs tables.

Run with: ``uv run --group benchmarks python -m benchmarks.tier3.report``.
Reads the JSONL records written by ``benchmarks.tier3.full`` and emits markdown tables
into ``docs/benchmarks/comparison/results/``. Per the published-values situation (see the
results page), Geo is reported as percentage gaps while Ran and Glover are reported as
matched/exceeded counts — their references are integer-quantized and partly loose, so
percentage gaps would over-read them.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.common.records import RunRecord, load_records
from benchmarks.mdplib.best_known import BestKnown, load_best_known

RECORDS_DIR = Path("reports/benchmarks/tier3")
RESULTS_DIR = Path("docs/benchmarks/comparison/results")

# Ladder rungs quoted in the tables (seconds; the last exists only for n=500 pairings).
TABLE_BUDGETS_SEC = (0.128, 1.024, 16.384)

# A best-of-seeds value within this relative tolerance of the reference counts as a match
# (references are printed with 4-6 significant digits).
MATCH_RTOL = 1e-4


def best_of_seeds(records: list[RunRecord], budget_sec: float) -> float | None:
    """Best MIN_SEPARATION over seeds at one ladder rung (None if the rung wasn't run)."""
    tag = f"time:{budget_sec}s"
    values = [r.quality["MIN_SEPARATION"] for r in records if r.budget == tag]
    return max(values) if values else None


def best_overall(records: list[RunRecord]) -> float | None:
    """Best MIN_SEPARATION over all seeds and all budgets run for an instance.

    The match/exceed counts use this rather than a fixed rung: instance sets run to
    different ladder depths (n=500 gets extended budget), and quoting a single low rung
    would hide the exceed cases that only the deeper budget surfaces.
    """
    values = [r.quality["MIN_SEPARATION"] for r in records if r.budget.startswith("time:")]
    return max(values) if values else None


def gap_pct(value: float, reference: float) -> float:
    """Gap to the reference in percent (positive = below the reference)."""
    return (reference - value) / reference * 100.0


def _group_records(records: list[RunRecord]) -> dict[tuple[str, int], list[RunRecord]]:
    """Group records by (instance name, k) — the published pairing key."""
    grouped: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for r in records:
        grouped[(r.problem, r.size)].append(r)
    return grouped


def build_geo_gap_table(rows: list[BestKnown], records: list[RunRecord]) -> str:
    """Markdown table for Geo: mean/worst gap per (n, k) group at the quoted budgets."""
    grouped = _group_records(records)
    by_nk: dict[tuple[int, int], list[BestKnown]] = defaultdict(list)
    for row in rows:
        if row.family == "Geo":
            by_nk[(row.n, row.k)].append(row)

    headers = " | ".join(f"gap @{b:g}s (mean / worst)" for b in TABLE_BUDGETS_SEC)
    lines = [f"| n | k | instances | {headers} |", "|---" * (3 + len(TABLE_BUDGETS_SEC)) + "|"]
    for (n, k), group in sorted(by_nk.items()):
        cells = []
        for budget in TABLE_BUDGETS_SEC:
            gaps = []
            for row in group:
                value = best_of_seeds(grouped[(row.instance, row.k)], budget)
                if value is not None:
                    gaps.append(gap_pct(value, row.best_known))
            cells.append("—" if not gaps else f"{np.mean(gaps):.1f}% / {max(gaps):.1f}%")
        lines.append(f"| {n} | {k} | {len(group)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_match_table(rows: list[BestKnown], records: list[RunRecord], family: str) -> str:
    """Markdown table for Ran/Glover: matched / exceeded / below counts per (n, k) group."""
    grouped = _group_records(records)
    by_nk: dict[tuple[int, int], list[BestKnown]] = defaultdict(list)
    for row in rows:
        if row.family == family:
            by_nk[(row.n, row.k)].append(row)

    lines = [
        "| n | k | instances | exceeded | matched | below (best over seeds & budgets) |",
        "|---" * 6 + "|",
    ]
    for (n, k), group in sorted(by_nk.items()):
        exceeded = matched = below = 0
        for row in group:
            value = best_overall(grouped[(row.instance, row.k)])
            if value is None:
                continue
            if value > row.best_known * (1 + MATCH_RTOL):
                exceeded += 1
            elif value >= row.best_known * (1 - MATCH_RTOL):
                matched += 1
            else:
                below += 1
        lines.append(f"| {n} | {k} | {len(group)} | {exceeded} | {matched} | {below} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Emit all tier-3 docs tables from the recorded runs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_best_known()
    records = load_records(RECORDS_DIR / "records.jsonl")

    (RESULTS_DIR / "tier3_geo.md").write_text(build_geo_gap_table(rows, records))
    (RESULTS_DIR / "tier3_ran.md").write_text(build_match_table(rows, records, "Ran"))
    (RESULTS_DIR / "tier3_glover.md").write_text(build_match_table(rows, records, "Glover"))
    print(f"tier-3 report emitted into {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
