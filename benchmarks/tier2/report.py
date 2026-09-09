"""Emit the tier-2 docs artifacts: one anytime chart per size with the best one-shot result marked, and the tables.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.report``.
Merges two record sources: max-div's records as measured by ``benchmarks.tier2.full`` or
``benchmarks.tier2.rerun`` (untracked, re-measured whenever the solver changes), and the entrant
records from the tracked reference file under `DATA_DIR`.

Each chart shows both max-div series, every entrant as a dot at its own measured time and quality,
and a dotted line at the best non-max-div result of that size. The tier page embeds each chart by
its size-derived name (`chart_name`), so no chart list is written; one table per size is written as
a snippet for the tier's tables page.
"""

import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.common.protocol import QUOTED_BUDGETS_SEC
from benchmarks.common.records import RunRecord, budget_sec, load_records
from benchmarks.figures import ReferenceLine, plot_anytime_curve
from benchmarks.runners.maxdiv_runner import maxdiv_tool_label

from .full import DATA_DIR, ENTRANT_FILE, MAXDIV_FILE, METRIC, N_WORKERS, OUTPUT_DIR, PROBLEM

RECORDS_DIR = OUTPUT_DIR
QUALITY_DECIMALS = 4  # decimals the tables print quality at; ties are judged at this precision
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


def series_medians(records: list[RunRecord], tool: str) -> dict[float, tuple[float, float]]:
    """Return, per wall-clock budget of one budget-series tool, the median over seeds of (quality, measured time)."""
    by_budget: dict[float, list[RunRecord]] = defaultdict(list)
    for r in records:
        if r.tool == tool and (budget := budget_sec(r.budget)) is not None:
            by_budget[budget].append(r)
    return {
        budget: (
            statistics.median(r.quality[METRIC.name] for r in rows),
            statistics.median(r.measured_sec for r in rows),
        )
        for budget, rows in sorted(by_budget.items())
    }


def size_table_rows(records: list[RunRecord]) -> list[tuple[str, float, float]]:
    """Return one size's table rows as (label, quality, time), best quality first, the shorter time first on a tie.

    max-div contributes one row per series and quoted budget, labeled with the budget; every entrant
    contributes its mean over seeds. One ordering over all rows is what lets a reader compare
    max-div's result at a budget with the one-shot tools directly. A tie is judged at the precision
    the table prints, so rows the reader sees as equal are ordered by time.
    """
    rows: list[tuple[str, float, float]] = []
    for tool in (maxdiv_tool_label(), maxdiv_tool_label(n_workers=N_WORKERS)):
        medians = series_medians(records, tool)
        for budget in QUOTED_BUDGETS_SEC:
            if budget in medians:
                rows.append((f"{tool} @ {budget:g} s", *medians[budget]))
    rows += [(tool, quality, time) for tool, (quality, time) in entrant_means(records).items()]
    return sorted(rows, key=lambda row: (-round(row[1], QUALITY_DECIMALS), row[2]))


def overtake_sentence(records: list[RunRecord]) -> str:
    """Return the sentence naming the budget at which each max-div series reaches the best one-shot result."""
    best = best_entrant(records)
    if best is None:
        return ""
    series = ((maxdiv_tool_label(), "one worker"), (maxdiv_tool_label(n_workers=N_WORKERS), f"{N_WORKERS} workers"))
    parts = []
    for tool, workers in series:
        budget = overtake_budget(records, tool, best[1])
        reached = f"at a budget of {budget:g} s" if budget is not None else f"not within {QUOTED_BUDGETS_SEC[1]:g} s"
        parts.append(f"{reached} with {workers}")
    return f"`max-div` reaches the best one-shot result ({best[0]}) {parts[0]} and {parts[1]}.\n"


def build_size_table(records: list[RunRecord], n: int) -> str:
    """Build one size's markdown snippet: every tool's quality and time in one ordering, then the overtake sentence."""
    size_records = [r for r in records if r.n == n]
    lines = ["| tool | quality (min separation) | time |", "|---|---|---|"]
    lines += [
        f"| {label} | {quality:.{QUALITY_DECIMALS}f} | {time:.3g} s |"
        for label, quality, time in size_table_rows(size_records)
    ]
    return "\n".join(lines) + "\n\n" + overtake_sentence(size_records)


def chart_name(n: int) -> str:
    """Return the image file name of one size's chart."""
    return f"tier2_{PROBLEM}_{n}_{METRIC.name.lower()}.webp"


def render_charts(records: list[RunRecord], sizes: list[int], images_dir: Path) -> list[str]:
    """Render one chart per size with the best-entrant line and return the written image names.

    The random baseline is left off the charts: its value sits so far below every tool that
    the differences between the tools would occupy a small fraction of the y-axis.
    """
    names = []
    for n in sizes:
        size_records = [r for r in records if r.n == n and r.tool != "random"]
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

    for n in sizes:
        (results_dir / f"tier2_size_{n}.md").write_text(build_size_table(records, n))
    render_charts(records, sizes, docs_dir / "images")
    print(f"tier-2 report emitted into {docs_dir}", flush=True)


if __name__ == "__main__":
    main()
