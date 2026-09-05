"""Run a tiny end-to-end validation of the tier-2 plumbing: runners, records, and the chart with its reference line.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.smoke``.
Deliberately minuscule (one size, two budgets, two seeds) — the run validates plumbing, not
performance; the published protocol lives in ``benchmarks.tier2.full``.
"""

from pathlib import Path

from benchmarks.common import build_problem, save_records
from benchmarks.figures import ReferenceLine, plot_anytime_curve
from benchmarks.runners import run_adapter, run_maxdiv_budget_series
from benchmarks.tier2.full import METRIC, PROBLEM, competitor_adapters
from benchmarks.tier2.report import best_competitor

OUTPUT_DIR = Path("reports/benchmarks/smoke")


def main() -> None:
    """Run the smoke scenario and write records + one figure."""
    problem = build_problem(PROBLEM, n=200, diversity_metric=METRIC)
    seeds = (0, 1)
    records = run_maxdiv_budget_series(problem, problem_name=PROBLEM, size=200, time_budgets_sec=[0.01, 0.1], seeds=seeds)
    records += run_maxdiv_budget_series(
        problem, problem_name=PROBLEM, size=200, time_budgets_sec=[1.0], seeds=seeds, n_workers=2
    )
    for adapter in competitor_adapters():
        records += run_adapter(adapter, problem, problem_name=PROBLEM, size=200, seeds=seeds)

    save_records(records, OUTPUT_DIR / "records.jsonl")
    best = best_competitor(records)
    plot_anytime_curve(
        records,
        metric_name=METRIC.name,
        path=OUTPUT_DIR / "anytime_u1_200.webp",
        title="smoke: U1 n=200 (validation only, not a published result)",
        reference_lines=(ReferenceLine(best[1], f"best one-shot result ({best[0]})"),) if best else (),
    )
    print(f"smoke OK: {len(records)} records -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
