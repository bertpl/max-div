"""Emit the tier-3 docs artifacts: gap-to-best-known charts per instance group, and the tables.

Run with: ``uv run --group benchmarks python -m benchmarks.tier3.report``.
Merges max-div's records (`RECORDS_DIR`) with the entrants' tracked reference records (`DATA_DIR`)
and the vendored best-known table.

Every record's quality is first turned into its gap to the instance's best-known value, in
percent; one chart per instance group (family, n, k) then aggregates the group's instances the way
a chart aggregates seeds — mean curve with a min/max band — with a dotted line at zero gap and each
entrant as a dot. The tables are written as snippets for the tier's tables page.
"""

import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from benchmarks.common.protocol import QUOTED_BUDGETS_SEC
from benchmarks.common.records import RunRecord, load_records
from benchmarks.figures import ReferenceLine, plot_anytime_curve
from benchmarks.mdplib.best_known import BestKnown, load_best_known
from benchmarks.runners.maxdiv_runner import budget_tag, maxdiv_tool_label
from .full import CHARTED_FAMILIES, DATA_DIR, ENTRANT_FILE, MAXDIV_FILE, METRIC, N_WORKERS, OUTPUT_DIR

RECORDS_DIR = OUTPUT_DIR
DOCS_DIR = Path("docs/benchmarks/third_party/head_to_head")
GAP_METRIC = "GAP_TO_BEST_KNOWN_PCT"
# A value within this relative tolerance of the reference counts as a match (references are
# printed with 4-6 significant digits).
MATCH_RTOL = 1e-4


def gap_pct(value: float, reference: float) -> float:
    """Return the gap to the reference in percent (positive = below the reference)."""
    return (reference - value) / reference * 100.0


def gap_records(records: list[RunRecord], references: dict[tuple[str, int], BestKnown]) -> list[RunRecord]:
    """Re-express every record's min separation as its gap to the pairing's best-known value."""
    return [
        replace(r, quality={GAP_METRIC: gap_pct(r.quality[METRIC.name], references[(r.problem, r.size)].best_known)})
        for r in records
        if (r.problem, r.size) in references
    ]


def group_pairings(rows: list[BestKnown]) -> dict[tuple[str, int, int], list[BestKnown]]:
    """Group the charted pairings by (family, n, k), in that order."""
    groups: dict[tuple[str, int, int], list[BestKnown]] = defaultdict(list)
    for row in rows:
        if row.family in CHARTED_FAMILIES:
            groups[(row.family, row.n, row.k)].append(row)
    return dict(sorted(groups.items()))


def best_value(records: list[RunRecord], instance: str, k: int, budget_tag: str | None = None) -> float | None:
    """Return the best min separation over seeds (and both series) for one pairing, at one budget or over all budgets."""
    values = [
        r.quality[METRIC.name]
        for r in records
        if r.problem == instance and r.size == k and (budget_tag is None or r.budget == budget_tag)
    ]
    return max(values) if values else None


def classify(value: float | None, reference: float) -> str | None:
    """Return 'exceeded', 'matched' or 'below' relative to the reference, or None without a value."""
    if value is None:
        return None
    if value > reference * (1 + MATCH_RTOL):
        return "exceeded"
    return "matched" if value >= reference * (1 - MATCH_RTOL) else "below"


def build_gap_table(gaps: list[RunRecord], groups: dict[tuple[str, int, int], list[BestKnown]]) -> str:
    """Build the markdown table: per group, max-div's mean and worst gap at the quoted budgets, for both series."""
    lo, hi = QUOTED_BUDGETS_SEC
    single, multi = maxdiv_tool_label(), maxdiv_tool_label(n_workers=N_WORKERS)
    headers = " | ".join(
        f"{label} @{budget:g} s (mean / worst)" for label in ("1 worker", f"{N_WORKERS} workers") for budget in (lo, hi)
    )
    lines = [f"| set | n | k | instances | {headers} |", "|---" * 8 + "|"]
    for (family, n, k), rows in groups.items():
        instances = {row.instance for row in rows}
        cells = []
        for tool in (single, multi):
            for budget in (lo, hi):
                values = [
                    r.quality[GAP_METRIC]
                    for r in gaps
                    if r.problem in instances and r.size == k and r.tool == tool and r.budget == budget_tag(budget)
                ]
                cells.append("—" if not values else f"{statistics.mean(values):.1f}% / {max(values):.1f}%")
        lines.append(f"| {family} | {n} | {k} | {len(rows)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_count_table(records: list[RunRecord], groups: dict[tuple[str, int, int], list[BestKnown]]) -> str:
    """Build the markdown table: per group, how many instances max-div exceeded, matched or fell below the best-known value at T_max."""
    hi = QUOTED_BUDGETS_SEC[1]
    lines = [f"| set | n | k | instances | exceeded | matched | below (best over seeds and series at {hi:g} s) |", "|---" * 7 + "|"]
    for (family, n, k), rows in groups.items():
        counts = {"exceeded": 0, "matched": 0, "below": 0}
        for row in rows:
            verdict = classify(best_value(records, row.instance, row.k, budget_tag(hi)), row.best_known)
            if verdict:
                counts[verdict] += 1
        lines.append(f"| {family} | {n} | {k} | {len(rows)} | {counts['exceeded']} | {counts['matched']} | {counts['below']} |")
    return "\n".join(lines) + "\n"


def build_entrant_table(gaps: list[RunRecord], groups: dict[tuple[str, int, int], list[BestKnown]]) -> str:
    """Build the markdown table: per group, each entrant's mean gap over its instances and seeds."""
    tools = sorted({r.tool for r in gaps if r.budget == "single-shot"})
    lines = ["| set | n | k | " + " | ".join(tools) + " |", "|---" * (len(tools) + 3) + "|"]
    for (family, n, k), rows in groups.items():
        instances = {row.instance for row in rows}
        cells = []
        for tool in tools:
            values = [r.quality[GAP_METRIC] for r in gaps if r.problem in instances and r.size == k and r.tool == tool]
            cells.append("—" if not values else f"{statistics.mean(values):.1f}%")
        lines.append(f"| {family} | {n} | {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def glover_sentence(records: list[RunRecord], rows: list[BestKnown]) -> str:
    """Return one sentence stating on how many Glover pairings max-div reached the published value."""
    glover = [row for row in rows if row.family == "Glover"]
    verdicts = [classify(best_value(records, row.instance, row.k), row.best_known) for row in glover]
    reached = sum(v in ("matched", "exceeded") for v in verdicts)
    measured = sum(v is not None for v in verdicts)
    return (
        f"On the Glover set (n ≤ 30), max-div reaches the published value on {reached} of the {measured} measured "
        f"pairings (best over seeds and budgets); the set is too small to discriminate and is not charted.\n"
    )


def build_best_known_table(rows: list[BestKnown]) -> str:
    """Build the markdown table of the vendored best-known values with their provenance."""
    lines = ["| set | instance | n | k | best-known | 2010 value | source | proven optimal |", "|---" * 8 + "|"]
    for row in rows:
        lines.append(
            f"| {row.family} | {row.instance.removesuffix('.txt')} | {row.n} | {row.k} | {row.best_known:g} "
            f"| {row.best_known_2010:g} | {row.source} | {'yes' if row.proven_optimal else ''} |"
        )
    return "\n".join(lines) + "\n"


def chart_name(family: str, n: int, k: int) -> str:
    """Return the image file name of one group's chart."""
    return f"tier3_{family.lower()}_{n}_{k}.webp"


def render_charts(gaps: list[RunRecord], groups: dict[tuple[str, int, int], list[BestKnown]], images_dir: Path) -> dict[str, list[str]]:
    """Render one gap chart per group and return the written image names per family."""
    written: dict[str, list[str]] = defaultdict(list)
    for (family, n, k), rows in groups.items():
        instances = {row.instance for row in rows}
        group_records = [r for r in gaps if r.problem in instances and r.size == k]
        if not any(r.tool.startswith("max-div") for r in group_records):
            continue
        name = chart_name(family, n, k)
        plot_anytime_curve(
            group_records,
            metric_name=GAP_METRIC,
            path=images_dir / name,
            title=f"{family} n={n} k={k} ({len(rows)} instances) — gap to best-known",
            reference_lines=(ReferenceLine(0.0, "best-known value"),),
            y_label="gap to best-known [%]",
        )
        written[family].append(name)
    return dict(written)


def main(records_dir: Path = RECORDS_DIR, docs_dir: Path = DOCS_DIR, data_dir: Path = DATA_DIR) -> None:
    """Emit every tier-3 docs artifact."""
    rows = load_best_known()
    references = {(row.instance, row.k): row for row in rows}
    entrants = load_records(data_dir / ENTRANT_FILE) if (data_dir / ENTRANT_FILE).exists() else []
    maxdiv = load_records(records_dir / MAXDIV_FILE) if (records_dir / MAXDIV_FILE).exists() else []
    gaps = gap_records(entrants + maxdiv, references)
    groups = group_pairings(rows)
    results_dir = docs_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "tier3_gaps.md").write_text(build_gap_table(gaps, groups))
    (results_dir / "tier3_counts.md").write_text(build_count_table(maxdiv, groups))
    (results_dir / "tier3_entrants.md").write_text(build_entrant_table(gaps, groups))
    (results_dir / "tier3_glover_sentence.md").write_text(glover_sentence(maxdiv, rows))
    (results_dir / "tier3_best_known.md").write_text(build_best_known_table(rows))
    written = render_charts(gaps, groups, docs_dir / "images")
    for family in CHARTED_FAMILIES:
        names = written.get(family, [])
        (results_dir / f"tier3_charts_{family.lower()}.md").write_text(
            "\n".join(f"![{name.removesuffix('.webp')}](./images/{name})" for name in names) + "\n"
        )
    print(f"tier-3 report emitted into {docs_dir}", flush=True)


if __name__ == "__main__":
    main()
