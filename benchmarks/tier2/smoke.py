"""Run a tiny end-to-end validation that exercises the budget-series runner, adapter runner, records, and figure wiring.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.smoke``.
Deliberately minuscule (one problem, short budget series, two seeds) — the run validates plumbing,
not performance; real runs are configured separately.
"""

from pathlib import Path

from benchmarks.adapters import (
    ApricotFacilityLocation,
    FpsampleFPS,
    GreedyMaxSum,
    KMedoidsFasterPAM,
    QcSelectorMaxMin,
    RandomBaseline,
    RdkitMaxMin,
    SelectionAdapter,
    SkmatterFPS,
)
from benchmarks.common import build_problem, save_records, time_budget_series
from benchmarks.figures import plot_anytime_curve
from benchmarks.runners import run_adapter, run_maxdiv_budget_series
from max_div.metrics import DiversityMetric

OUTPUT_DIR = Path("reports/benchmarks/smoke")


def main() -> None:
    """Run the smoke scenario and write records + one figure."""
    problem = build_problem("U1", n=200, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION)
    seeds = (0, 1)

    records = run_maxdiv_budget_series(
        problem,
        problem_name="U1",
        size=200,
        time_budgets_sec=time_budget_series(0.001, 0.064),
        iteration_budgets=[100, 1000],
        seeds=seeds,
    )
    adapters: list[SelectionAdapter] = [
        RandomBaseline(),
        FpsampleFPS(),
        SkmatterFPS(),
        RdkitMaxMin(),
        ApricotFacilityLocation(),
        KMedoidsFasterPAM(),
        GreedyMaxSum(),
        QcSelectorMaxMin(),  # GPL opt-in group; skipped below when not installed
    ]
    for adapter in adapters:
        try:
            records += run_adapter(adapter, problem, problem_name="U1", size=200, seeds=seeds)
        except ImportError:
            print(f"skipped (not installed): {adapter.name}")

    save_records(records, OUTPUT_DIR / "records.jsonl")
    plot_anytime_curve(
        [r for r in records if not r.budget.startswith("iterations:")],
        metric_name=DiversityMetric.GEOMEAN_SEPARATION.name,
        path=OUTPUT_DIR / "anytime_u1_s2.webp",
        title="smoke: U1 n=200 (validation only, not a published result)",
    )
    print(f"smoke OK: {len(records)} records -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
