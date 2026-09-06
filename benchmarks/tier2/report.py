"""Emit the tier-2 docs artifacts: one anytime chart per size with the best one-shot result marked, and the tables.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.report``.
Merges two record sources: max-div's records as measured by ``benchmarks.tier2.full`` or
``benchmarks.tier2.rerun`` (untracked, re-measured whenever the solver changes), and the entrant
records from the tracked reference file under `DATA_DIR`.

Each chart shows both max-div series, every entrant as a dot at its own measured time and quality,
and a dotted line at the best non-max-div result of that size. The tier page embeds each chart by
its size-derived name (`chart_name`), so no chart list is written; the tables are written as
snippets for the tier's tables page.
"""

import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.common.protocol import QUOTED_BUDGETS_SEC
from benchmarks.common.records import RunRecord, load_records
from benchmarks.figures import ReferenceLine, plot_anytime_curve
from benchmarks.runners.maxdiv_runner import budget_sec, maxdiv_tool_label
from .full import DATA_DIR, ENTRANT_FILE, MAXDIV_FILE, METRIC, N_WORKERS, OUTPUT_DIR, PROBLEM

RECORDS_DIR = OUTPUT_DIR
DOCS_DIR = Path("docs/benchmarks/third_party/head_to_head")


def is_entrant(record: RunRecord) -> bool:
    """Return whether a record is an entrant's: any single-shot tool except the random baseline, which marks the quality of an unoptimized selection."""
    return record.budget == "single-shot" and record.tool != "random"


def entrant_means(records: list[RunRecord]) -> dict[str, tuple[float, float]]:
    """Return, per entrant, the mean over seeds of (quality, measured time) at one size."""
    by_tool: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        if is_entrant(r):
            by_tool[r.tool].append(r)
    return {
        tool: (statistics.mean(r.quality[METRIC.name] for r in rows), statistics.mean(r.measured_sec for r in rows))
        for tool, rows in by_tool.items()
    }


def best_entrant(records: list[RunRecord]) -> tuple[str, float, float] | None:
    """Return the entrant with the highest mean quality at one size as (tool, quality, time), or None without entrants."""
    means = entrant_means(records)
    if not means:
        return None
    tool = max(means, key=lambda t: means[t][0])
    return tool, means[tool][0], means[tool][1]


def median_by_budget(records: list[RunRecord], tool: str) -> dict[float, float]:
    """Return the median quality over seeds per wall-clock budget of one budget-series tool, keyed by budget in seconds."""
    by_budget: dict[float, list[float]] = defaultdict(list)
    for r in records:
        if r.tool == tool and (budget := budget_sec(r.budget)) is not None:
            by_budget[budget].append(r.quality[METRIC.name])
    return {budget: statistics.median(values) for budget, values in sorted(by_budget.items())}


def overtake_budget(records: list[RunRecord], tool: str, target: float) -> float | None:
    """Return the smallest budget at which the tool's median quality reaches `target`, or None if it never does."""
    return next((budget for budget, median in median_by_budget(records, tool).items() if median >= target), None)


def build_summary_table(records: list[RunRecord], sizes: list[int]) -> str:
    """Build the markdown table: per size, the best entrant, max-div's medians at the quoted budgets, and the overtake budgets."""
    lo, hi = QUOTED_BUDGETS_SEC
    single, multi = maxdiv_tool_label(), maxdiv_tool_label(n_workers=N_WORKERS)
    lines = [
        f"| n | best one-shot tool | its quality | its time | 1 worker @{lo:g} s | 1 worker @{hi:g} s "
        f"| {N_WORKERS} workers @{lo:g} s | {N_WORKERS} workers @{hi:g} s | overtake budget, 1 worker | overtake budget, {N_WORKERS} workers |",
        "|---" * 10 + "|",
    ]
    for n in sizes:
        size_records = [r for r in records if r.n == n]
        best = best_entrant(size_records)
        best_cells = f"{best[0]} | {best[1]:.4f} | {best[2]:.3g} s" if best else "— | — | —"
        medians = [median_by_budget(size_records, tool).get(budget) for tool in (single, multi) for budget in (lo, hi)]
        median_cells = " | ".join("—" if m is None else f"{m:.4f}" for m in medians)
        overtakes = [overtake_budget(size_records, tool, best[1]) if best else None for tool in (single, multi)]
        overtake_cells = " | ".join("—" if b is None else f"{b:g} s" for b in overtakes)
        lines.append(f"| {n:,} | {best_cells} | {median_cells} | {overtake_cells} |")
    return "\n".join(lines) + "\n"


def build_entrant_table(records: list[RunRecord], sizes: list[int]) -> str:
    """Build the markdown table: per size, every entrant's mean quality and mean time; a dash where a tool did not run."""
    tools = sorted({r.tool for r in records if is_entrant(r)})
    lines = ["| tool | " + " | ".join(f"n = {n:,}" for n in sizes) + " |", "|---" * (len(sizes) + 1) + "|"]
    means_by_size = {n: entrant_means([r for r in records if r.n == n]) for n in sizes}
    for tool in tools:
        cells = [
            f"{means_by_size[n][tool][0]:.4f} ({means_by_size[n][tool][1]:.3g} s)" if tool in means_by_size[n] else "—"
            for n in sizes
        ]
        lines.append(f"| {tool} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def chart_name(n: int) -> str:
    """Return the image file name of one size's chart."""
    return f"tier2_{PROBLEM}_{n}_{METRIC.name.lower()}.webp"


def render_charts(records: list[RunRecord], sizes: list[int], images_dir: Path) -> list[str]:
    """Render one chart per size with the best-entrant line and return the written image names."""
    names = []
    for n in sizes:
        size_records = [r for r in records if r.n == n]
        if not any(r.tool.startswith("max-div") for r in size_records):
            continue
        best = best_entrant(size_records)
        lines = (ReferenceLine(best[1], f"best one-shot result ({best[0]})"),) if best else ()
        name = chart_name(n)
        plot_anytime_curve(
            size_records,
            metric_name=METRIC.name,
            path=images_dir / name,
            title=f"{PROBLEM} (n={n:,}) — {METRIC.name}",
            reference_lines=lines,
        )
        names.append(name)
    return names


def main(records_dir: Path = RECORDS_DIR, docs_dir: Path = DOCS_DIR, data_dir: Path = DATA_DIR) -> None:
    """Emit every tier-2 docs artifact from the merged record sources."""
    records = load_records(data_dir / ENTRANT_FILE) + load_records(records_dir / MAXDIV_FILE)
    sizes = sorted({r.n for r in records})
    results_dir = docs_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "tier2_summary.md").write_text(build_summary_table(records, sizes))
    (results_dir / "tier2_entrants.md").write_text(build_entrant_table(records, sizes))
    render_charts(records, sizes, docs_dir / "images")
    print(f"tier-2 report emitted into {docs_dir}", flush=True)


if __name__ == "__main__":
    main()
