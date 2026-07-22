"""The published tier-2 comparison run: max-div vs. Python subset-selection heuristics.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.full``.
Emits one JSONL record per measurement into ``reports/benchmarks/tier2/``; figures and
tables for the docs are generated separately by ``benchmarks.tier2.report`` from those
records. Expect a multi-hour sequential run on a quiet machine.

Protocol (mirrored on the results page):

- Unconstrained: U1-U4 at sizes 2/10/50/200 (n = 200 to 20000). max-div runs the wall-clock
  budget ladder once per diversity metric (it optimizes what it is scored on); single-shot
  competitors run once per seed and are scored under every metric alike.
- Constrained: C1-C2 at sizes 2/10/20 (n <= 2000) with MIN_SEPARATION, against code-FDM —
  the one surveyed heuristic that honors group constraints. It does not scale past n ~ 2000
  (capping the sizes), and its disjoint-color fairness model cannot express the overlapping
  constraint groups of C3/C4 (excluding those problems).
"""

from pathlib import Path

from benchmarks.adapters import (
    ApricotFacilityLocation,
    CodeFdmFairFlow,
    FpsampleFPS,
    GreedyMaxSum,
    KMedoidsFasterPAM,
    QcSelectorMaxMin,
    RandomBaseline,
    RdkitMaxMin,
    SelectionAdapter,
    SkmatterFPS,
)
from benchmarks.common import build_problem, save_records, time_ladder
from benchmarks.common.records import RunRecord
from benchmarks.runners import run_adapter, run_maxdiv_ladder
from max_div.metrics import DiversityMetric

OUTPUT_DIR = Path("reports/benchmarks/tier2")

SEEDS = (0, 1, 2)
UNCONSTRAINED_PROBLEMS = ("U1", "U2", "U3", "U4")
UNCONSTRAINED_SIZES = (2, 10, 50, 200)
# C3/C4 are excluded: their overlapping constraint groups are inexpressible in code-FDM's
# disjoint-color model (the adapter refuses them), leaving no competitor to compare against.
CONSTRAINED_PROBLEMS = ("C1", "C2")
CONSTRAINED_SIZES = (2, 10, 20)
CONSTRAINED_METRIC = DiversityMetric.MIN_SEPARATION

# Ladder ceiling 10 s: the last rung is the first value >= the ceiling (16.4 s), so the
# curves bracket the region where the slowest competitors land instead of stopping short.
TIME_BUDGETS_SEC = time_ladder(0.001, 10.0)

EVALUATED_DIVERSITY_METRICS = (
    DiversityMetric.MIN_SEPARATION,
    DiversityMetric.MEAN_SEPARATION,
    DiversityMetric.GEOMEAN_SEPARATION,
    DiversityMetric.MEAN_PAIRWISE_DISTANCE,
)


def unconstrained_adapters() -> list[SelectionAdapter]:
    """The tier-2 competitor roster (GPL-group qc-selector included only when installed)."""
    return [
        RandomBaseline(),
        FpsampleFPS(),
        SkmatterFPS(),
        RdkitMaxMin(),
        ApricotFacilityLocation(),
        KMedoidsFasterPAM(),
        GreedyMaxSum(),
        QcSelectorMaxMin(),
    ]


def run_unconstrained() -> list[RunRecord]:
    """All unconstrained measurements: competitor single-shots + max-div ladder per metric."""
    records: list[RunRecord] = []
    for name in UNCONSTRAINED_PROBLEMS:
        for size in UNCONSTRAINED_SIZES:
            problem = build_problem(name, size=size, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION)
            for adapter in unconstrained_adapters():
                try:
                    records += run_adapter(adapter, problem, problem_name=name, size=size, seeds=SEEDS)
                except ImportError:
                    print(f"  skipped (not installed): {adapter.name}", flush=True)
            for metric in EVALUATED_DIVERSITY_METRICS:
                metric_problem = build_problem(name, size=size, diversity_metric=metric)
                records += run_maxdiv_ladder(
                    metric_problem,
                    problem_name=name,
                    size=size,
                    time_budgets_sec=TIME_BUDGETS_SEC,
                    seeds=SEEDS,
                )
            print(f"{name} size={size} done ({len(records)} records so far)", flush=True)
            save_records(records, OUTPUT_DIR / "records_unconstrained.jsonl")
    return records


def run_constrained() -> list[RunRecord]:
    """All constrained measurements: max-div ladder vs. code-FDM, MIN_SEPARATION."""
    records: list[RunRecord] = []
    for name in CONSTRAINED_PROBLEMS:
        for size in CONSTRAINED_SIZES:
            problem = build_problem(name, size=size, diversity_metric=CONSTRAINED_METRIC)
            records += run_adapter(CodeFdmFairFlow(), problem, problem_name=name, size=size, seeds=SEEDS)
            records += run_maxdiv_ladder(
                problem,
                problem_name=name,
                size=size,
                time_budgets_sec=TIME_BUDGETS_SEC,
                seeds=SEEDS,
            )
            print(f"{name} size={size} done ({len(records)} records so far)", flush=True)
            save_records(records, OUTPUT_DIR / "records_constrained.jsonl")
    return records


def main() -> None:
    """Run the full tier-2 scenario and persist all records."""
    print("tier-2 unconstrained ...", flush=True)
    unconstrained = run_unconstrained()
    print("tier-2 constrained ...", flush=True)
    constrained = run_constrained()
    print(f"tier-2 complete: {len(unconstrained)} unconstrained + {len(constrained)} constrained records", flush=True)


if __name__ == "__main__":
    main()
