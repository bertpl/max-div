"""Tier-2 report emission: turn the recorded runs into docs figures and tables.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.report``.
Merges two record sources: max-div's ladders as measured by ``benchmarks.tier2.full`` or
``benchmarks.tier2.rerun`` (untracked, re-measured whenever the solver changes), and the
competitor single-shots from the tracked reference records in ``benchmarks/tier2/data/``
(fixed across max-div re-measurements). Emits the curated docs artifacts (anytime-curve
figures + margin tables) into ``docs/benchmarks/comparison/``.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.common.records import RunRecord, load_records
from benchmarks.figures import plot_anytime_curve

RECORDS_DIR = Path("reports/benchmarks/tier2")
DATA_DIR = Path(__file__).parent / "data"
DOCS_DIR = Path("docs/benchmarks/comparison")

# Figures are curated to one representative problem per scenario (U1 uniform / C1 simple
# constraints); the margin tables cover every problem, size, and metric.
FIGURE_UNCONSTRAINED_PROBLEM = "U1"
FIGURE_UNCONSTRAINED_METRICS = ("MIN_SEPARATION", "GEOMEAN_SEPARATION")
FIGURE_CONSTRAINED_PROBLEM = "C1"
TABLE_METRICS = ("MIN_SEPARATION", "MEAN_SEPARATION", "GEOMEAN_SEPARATION", "MEAN_PAIRWISE_DISTANCE")

# Budget rungs the margin tables quote (seconds; matched to the nearest ladder rung).
TABLE_BUDGETS_SEC = (1.024, 16.384)


def records_for_figure(records: list[RunRecord], metric_name: str) -> list[RunRecord]:
    """Select the records one figure shows: every competitor + the max-div run optimizing the metric."""
    return [r for r in records if not r.tool.startswith("max-div") or r.diversity_metric == metric_name]


def maxdiv_mean_at_budget(records: list[RunRecord], metric_name: str, budget_sec: float) -> float | None:
    """Mean quality over seeds of the max-div run optimizing the metric, at the given ladder rung."""
    tag = f"time:{budget_sec}s"
    values = [
        r.quality[metric_name]
        for r in records
        if r.tool.startswith("max-div") and r.diversity_metric == metric_name and r.budget == tag
    ]
    return float(np.mean(values)) if values else None


def best_competitor_mean(records: list[RunRecord], metric_name: str) -> tuple[str, float] | None:
    """Best per-tool mean quality (over seeds) among all single-shot competitors."""
    by_tool: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if not r.tool.startswith("max-div"):
            by_tool[r.tool].append(r.quality[metric_name])
    if not by_tool:
        return None
    means = {tool: float(np.mean(vals)) for tool, vals in by_tool.items()}
    best = max(means, key=means.get)
    return best, means[best]


def margin_pct(records: list[RunRecord], metric_name: str, budget_sec: float) -> float | None:
    """max-div's advantage over the best competitor, in percent (positive = max-div wins)."""
    ours = maxdiv_mean_at_budget(records, metric_name, budget_sec)
    best = best_competitor_mean(records, metric_name)
    if ours is None or best is None:
        return None
    return (ours - best[1]) / best[1] * 100.0


def build_margin_table(records: list[RunRecord], metric_name: str, problems: list[str], sizes: list[int]) -> str:
    """Markdown margin table for one metric: rows = problem size, one column per problem.

    Each cell holds max-div's margin vs. the best competitor at the two quoted budgets;
    the last column names the best competitor (evaluated at the largest size).
    """
    by_key: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for r in records:
        by_key[(r.problem, r.size)].append(r)

    b_lo, b_hi = TABLE_BUDGETS_SEC
    lines = [
        "| n | " + " | ".join(problems) + " |",
        "|---" * (len(problems) + 1) + "|",
    ]
    for size in sizes:
        cells = []
        n_value = 0
        for problem in problems:
            recs = by_key.get((problem, size), [])
            n_value = recs[0].n if recs else n_value
            lo, hi = margin_pct(recs, metric_name, b_lo), margin_pct(recs, metric_name, b_hi)
            cells.append("-" if lo is None else f"{lo:+.1f}% / {hi:+.1f}%")
        lines.append(f"| {n_value} | " + " | ".join(cells) + " |")
    header = (
        f"*max-div's margin vs. the best single-shot competitor under `{metric_name}` "
        f"(positive = max-div ahead), at ~{b_lo:.0f} s / ~{b_hi:.0f} s of budget:*\n"
    )
    return header + "\n" + "\n".join(lines) + "\n"


def main(records_dir: Path = RECORDS_DIR, docs_dir: Path = DOCS_DIR) -> None:
    """Emit all curated tier-2 docs artifacts from the merged record sources."""
    unconstrained = load_records(DATA_DIR / "third_party_unconstrained.jsonl") + load_records(
        records_dir / "maxdiv_unconstrained.jsonl"
    )
    constrained = load_records(DATA_DIR / "third_party_constrained.jsonl") + load_records(
        records_dir / "maxdiv_constrained.jsonl"
    )
    images_dir = docs_dir / "images"
    results_dir = docs_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sizes = sorted({r.size for r in unconstrained})
    problems = sorted({r.problem for r in unconstrained})
    for metric_name in FIGURE_UNCONSTRAINED_METRICS:
        for size in sizes:
            recs = [r for r in unconstrained if r.problem == FIGURE_UNCONSTRAINED_PROBLEM and r.size == size]
            n = recs[0].n
            plot_anytime_curve(
                records_for_figure(recs, metric_name),
                metric_name=metric_name,
                path=images_dir / f"tier2_{FIGURE_UNCONSTRAINED_PROBLEM}_{size}_{metric_name.lower()}.svg",
                title=f"{FIGURE_UNCONSTRAINED_PROBLEM} (n={n}) — {metric_name}",
            )

    for metric_name in TABLE_METRICS:
        table = build_margin_table(unconstrained, metric_name, problems, sizes)
        (results_dir / f"tier2_margins_{metric_name.lower()}.md").write_text(table)

    constrained_sizes = sorted({r.size for r in constrained})
    for size in constrained_sizes:
        recs = [r for r in constrained if r.problem == FIGURE_CONSTRAINED_PROBLEM and r.size == size]
        n = recs[0].n
        plot_anytime_curve(
            records_for_figure(recs, "MIN_SEPARATION"),
            metric_name="MIN_SEPARATION",
            path=images_dir / f"tier2_{FIGURE_CONSTRAINED_PROBLEM}_{size}_min_separation.svg",
            title=f"{FIGURE_CONSTRAINED_PROBLEM} constrained (n={n}) — MIN_SEPARATION",
        )
    table = build_margin_table(
        constrained, "MIN_SEPARATION", sorted({r.problem for r in constrained}), constrained_sizes
    )
    (results_dir / "tier2_margins_constrained.md").write_text(table)

    print(f"tier-2 report emitted into {docs_dir}", flush=True)


if __name__ == "__main__":
    main()
