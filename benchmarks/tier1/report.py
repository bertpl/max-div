"""Emit the tier-1 docs artifacts: anytime charts against certified optima, and the gap and certification tables.

Run with: ``uv run --group benchmarks python -m benchmarks.tier1.report``.
Merges two result sources: max-div's records as measured by ``benchmarks.tier1.full`` or
``benchmarks.tier1.rerun`` (untracked, re-measured whenever the solver changes), and the exact
solvers' certified optima from the tracked files under `DATA_DIR`.

Per certified (problem, size, objective) cell one chart is written under the docs images folder:
both max-div series, a dotted line at the certified optimum, and one marker per certifying solver
at (proof time, optimum). `FULL_WIDTH_OBJECTIVE`'s charts are listed full width in one snippet per
problem; every other objective gets one thumbnail-gallery snippet. The tables are written as snippets for
the tier's tables page.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.common.protocol import QUOTED_BUDGETS_SEC
from benchmarks.common.records import RunRecord, load_records
from benchmarks.common.registry import display_name
from benchmarks.figures import ReferenceLine, ReferenceMarker, plot_anytime_curve
from benchmarks.figures.style import tool_color
from benchmarks.runners.maxdiv_runner import budget_tag, maxdiv_tool_label
from .full import DATA_DIR, EXACT_MAXMIN_FILE, EXACT_NN_FILE, N_WORKERS, OUTPUT_DIR, PROBLEMS, maxdiv_records_path
from max_div.metrics import DiversityMetric

RECORDS_DIR = OUTPUT_DIR
DOCS_DIR = Path("docs/benchmarks/third_party/head_to_head")
OBJECTIVES = (DiversityMetric.MIN_SEPARATION, DiversityMetric.MEAN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION)
FULL_WIDTH_OBJECTIVE = DiversityMetric.MIN_SEPARATION  # the other objectives get thumbnail galleries


def median_quality(records: list[RunRecord], tool: str, metric_name: str, budget_sec: float) -> float | None:
    """Return the median quality over seeds of one tool at one budget, or None when that budget was not run."""
    values = [r.quality[metric_name] for r in records if r.tool == tool and r.budget == budget_tag(budget_sec)]
    return statistics.median(values) if values else None


def gap_pct(value: float | None, optimum: float) -> float | None:
    """Return the gap to the certified optimum in percent (positive = below the optimum), or None without a value."""
    return None if value is None else (optimum - value) / optimum * 100.0


def format_gap(gap: float | None) -> str:
    """Format a gap cell to one decimal; a gap that rounds to zero prints as 0.0 % whatever its sign.

    max-div's float32 arithmetic lands a hair above or below the certified optimum on the cells it
    solves exactly, which would otherwise print as -0.0 %.
    """
    if gap is None:
        return "—"
    return f"{0.0 if abs(gap) < 0.05 else gap:.1f}%"


def certified_optima(exact_rows: list[dict]) -> dict[tuple[str, str, int], list[dict]]:
    """Group the certifying rows per (problem, objective, n); every row in a group agrees on the optimum."""
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in exact_rows:
        if row["proven_optimal"]:
            groups[(row["problem"], row["objective"], row["n"])].append(row)
    return dict(groups)


def build_gap_table(exact_rows: list[dict], records: list[RunRecord], metric: DiversityMetric) -> str:
    """Build the markdown table for one objective: per certified cell, the optimum, who certified it, and max-div's gaps.

    The gap columns quote the median over seeds at the protocol's quoted budgets, for the
    single-worker and the multi-worker series.
    """
    lo, hi = QUOTED_BUDGETS_SEC
    single, multi = maxdiv_tool_label(), maxdiv_tool_label(n_workers=N_WORKERS)
    lines = [
        f"| problem | n | k | certified optimum | certified by | gap 1 worker @{lo:g} s | gap 1 worker @{hi:g} s "
        f"| gap {N_WORKERS} workers @{lo:g} s | gap {N_WORKERS} workers @{hi:g} s |",
        "|---" * 9 + "|",
    ]
    for (problem, objective, n), rows in sorted(certified_optima(exact_rows).items()):
        if objective != metric.name:
            continue
        optimum = rows[0]["optimum"]
        cell_records = [r for r in records if r.problem == problem and r.n == n]
        certifiers = ", ".join(f"{display_name(r['solver'])} ({r['measured_sec']:.1f} s)" for r in rows)
        gaps = [
            gap_pct(median_quality(cell_records, tool, metric.name, budget), optimum)
            for tool in (single, multi)
            for budget in (lo, hi)
        ]
        cells = " | ".join(format_gap(g) for g in gaps)
        lines.append(f"| {problem} | {n} | {rows[0]['k']} | {optimum:.4f} | {certifiers} | {cells} |")
    return "\n".join(lines) + "\n"


def build_certification_table(exact_rows: list[dict]) -> str:
    """Build the markdown table: per solver, problem and objective, the largest certified n and where certification stopped."""
    by_column: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in exact_rows:
        by_column[(row["solver"], row["problem"], row["objective"])].append(row)
    lines = ["| solver | problem | objective | largest certified n | certification stopped at |", "|---" * 5 + "|"]
    for (solver, problem, objective), rows in sorted(by_column.items()):
        certified = [r["n"] for r in rows if r["proven_optimal"]]
        failed = [r for r in rows if not r["proven_optimal"]]
        largest = f"**{max(certified):,}**" if certified else "—"
        stopped = f"n={failed[0]['n']:,} ({failed[0]['measured_sec']:.0f} s, not certified)" if failed else "grid exhausted"
        lines.append(f"| {display_name(solver)} | {problem} | {objective} | {largest} | {stopped} |")
    return "\n".join(lines) + "\n"


def chart_name(problem: str, n: int, metric: DiversityMetric) -> str:
    """Return the image file name of one cell's chart."""
    return f"tier1_{problem}_{n}_{metric.name.lower()}.webp"


def render_charts(
    exact_rows: list[dict], records_by_metric: dict[str, list[RunRecord]], images_dir: Path
) -> dict[tuple[str, str], list[str]]:
    """Render one chart per certified cell and return the written image names per (objective, problem)."""
    written: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (problem, objective, n), rows in sorted(certified_optima(exact_rows).items()):
        metric = DiversityMetric[objective]
        cell_records = [r for r in records_by_metric.get(objective, []) if r.problem == problem and r.n == n]
        if not cell_records:
            continue
        optimum = rows[0]["optimum"]
        markers = tuple(
            ReferenceMarker(r["measured_sec"], optimum, f"{display_name(r['solver'])} proof", color=tool_color(r["solver"]))
            for r in rows
        )
        name = chart_name(problem, n, metric)
        plot_anytime_curve(
            cell_records,
            metric_name=metric.name,
            path=images_dir / name,
            title=f"{problem} (n={n}) — {metric.name}",
            reference_lines=(ReferenceLine(optimum, "certified optimum"),),
            reference_markers=markers,
        )
        written[(objective, problem)].append(name)
    return dict(written)


def full_width_snippet(names: list[str]) -> str:
    """Return markdown listing every chart full width, one per line."""
    return "\n".join(f"![{name.removesuffix('.webp')}](../images/{name})" for name in names) + "\n"


def gallery_snippet(names: list[str], metric: DiversityMetric) -> str:
    """Return an HTML thumbnail gallery, three per row, each linking to its full-size chart (the page dir sits one level below `images/`)."""
    thumbnails = [
        f'<a href="../images/{name}"><img src="../images/{name}" alt="{metric.name} anytime chart {name}" width="32%"></a>'
        for name in names
    ]
    return " ".join(thumbnails) + "\n"


def main(records_dir: Path = RECORDS_DIR, docs_dir: Path = DOCS_DIR, data_dir: Path = DATA_DIR) -> None:
    """Emit every tier-1 docs artifact from the merged result sources."""
    exact_rows = json.loads((data_dir / EXACT_MAXMIN_FILE).read_text()) + json.loads((data_dir / EXACT_NN_FILE).read_text())
    records_by_metric = {
        metric.name: load_records(maxdiv_records_path(metric, records_dir))
        for metric in OBJECTIVES
        if maxdiv_records_path(metric, records_dir).exists()
    }
    results_dir = docs_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir = docs_dir / "images"

    for metric in OBJECTIVES:
        table = build_gap_table(exact_rows, records_by_metric.get(metric.name, []), metric)
        (results_dir / f"tier1_gap_{metric.name.lower()}.md").write_text(table)
    (results_dir / "tier1_certification.md").write_text(build_certification_table(exact_rows))

    written = render_charts(exact_rows, records_by_metric, images_dir)
    for metric in OBJECTIVES:
        if metric == FULL_WIDTH_OBJECTIVE:
            # one snippet per problem: the page gives each problem its own section
            for problem in PROBLEMS:
                names = written.get((metric.name, problem), [])
                snippet = results_dir / f"tier1_charts_{metric.name.lower()}_{problem.lower()}.md"
                snippet.write_text(full_width_snippet(names))
        else:
            names = [name for problem in PROBLEMS for name in written.get((metric.name, problem), [])]
            (results_dir / f"tier1_gallery_{metric.name.lower()}.md").write_text(gallery_snippet(names, metric))
    print(f"tier-1 report emitted into {docs_dir}", flush=True)


if __name__ == "__main__":
    main()
