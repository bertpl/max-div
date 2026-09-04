"""The published tier-2 comparison run: max-div vs. Python subset-selection heuristics.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.full``.
Emits one JSONL record file per (origin, scenario) into ``reports/benchmarks/tier2/`` —
competitor and max-div records are kept in separate files so max-div can be re-measured
alone (``benchmarks.tier2.rerun``) while the competitor side stays fixed. Figures and
tables for the docs are generated separately by ``benchmarks.tier2.report``. Expect a
multi-hour sequential run on a quiet machine.

The competitor outputs (``third_party_*.jsonl``) share their names with the tracked
reference copies under ``benchmarks/tier2/data/``: promoting a fresh competitor
measurement means copying the files over by hand, so a re-run never changes the tracked
references by itself.

Protocol (mirrored on the results page):

- Unconstrained: U1-U4 at sizes 2/10/50/200 (n = 200 to 20000). max-div runs the wall-clock
  budget series once per diversity metric (it optimizes what it is scored on); single-shot
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
from benchmarks.common import build_problem, save_records, time_budget_series
from benchmarks.common.records import RunRecord
from benchmarks.runners import run_adapter, run_maxdiv_budget_series
from max_div.metrics import DiversityMetric

OUTPUT_DIR = Path("reports/benchmarks/tier2")

SEEDS = (0, 1, 2)
UNCONSTRAINED_PROBLEMS = ("U1", "U2", "U3", "U4")
UNCONSTRAINED_SIZES = (200, 1000, 5000, 20000)
# C3/C4 are excluded: their overlapping constraint groups are inexpressible in code-FDM's
# disjoint-color model (the adapter refuses them), leaving no competitor to compare against.
CONSTRAINED_PROBLEMS = ("C1", "C2")
CONSTRAINED_SIZES = (200, 1000, 2000)
CONSTRAINED_METRIC = DiversityMetric.MIN_SEPARATION

# The series ends at the first budget >= 10 s (16.4 s), so the curves extend past the
# region where the slowest competitors land.
TIME_BUDGETS_SEC = time_budget_series(0.001, 10.0)

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


def run_competitors_unconstrained(
    out_path: Path = OUTPUT_DIR / "third_party_unconstrained.jsonl",
) -> list[RunRecord]:
    """All unconstrained competitor single-shots, one run per seed."""
    records: list[RunRecord] = []
    for name in UNCONSTRAINED_PROBLEMS:
        for size in UNCONSTRAINED_SIZES:
            problem = build_problem(name, n=size, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION)
            for adapter in unconstrained_adapters():
                try:
                    records += run_adapter(adapter, problem, problem_name=name, size=size, seeds=SEEDS)
                except ImportError:
                    print(f"  skipped (not installed): {adapter.name}", flush=True)
            print(f"competitors {name} size={size} done ({len(records)} records so far)", flush=True)
            save_records(records, out_path)
    return records


def run_maxdiv_unconstrained(
    problems: tuple[str, ...] = UNCONSTRAINED_PROBLEMS,
    sizes: tuple[int, ...] = UNCONSTRAINED_SIZES,
    time_budgets_sec: list[float] = TIME_BUDGETS_SEC,
    seeds: tuple[int, ...] = SEEDS,
    out_path: Path = OUTPUT_DIR / "maxdiv_unconstrained.jsonl",
) -> list[RunRecord]:
    """max-div's budget series on every unconstrained cell, one series per diversity metric.

    Defaults are the published protocol; pass smaller values only for validation runs.
    """
    records: list[RunRecord] = []
    for name in problems:
        for size in sizes:
            for metric in EVALUATED_DIVERSITY_METRICS:
                metric_problem = build_problem(name, n=size, diversity_metric=metric)
                records += run_maxdiv_budget_series(
                    metric_problem,
                    problem_name=name,
                    size=size,
                    time_budgets_sec=time_budgets_sec,
                    seeds=seeds,
                )
            print(f"max-div {name} size={size} done ({len(records)} records so far)", flush=True)
            save_records(records, out_path)
    return records


def run_competitors_constrained(
    out_path: Path = OUTPUT_DIR / "third_party_constrained.jsonl",
) -> list[RunRecord]:
    """code-FDM single-shots on the constrained cells, one run per seed."""
    records: list[RunRecord] = []
    for name in CONSTRAINED_PROBLEMS:
        for size in CONSTRAINED_SIZES:
            problem = build_problem(name, n=size, diversity_metric=CONSTRAINED_METRIC)
            records += run_adapter(CodeFdmFairFlow(), problem, problem_name=name, size=size, seeds=SEEDS)
            print(f"competitors {name} size={size} done ({len(records)} records so far)", flush=True)
            save_records(records, out_path)
    return records


def run_maxdiv_constrained(
    problems: tuple[str, ...] = CONSTRAINED_PROBLEMS,
    sizes: tuple[int, ...] = CONSTRAINED_SIZES,
    time_budgets_sec: list[float] = TIME_BUDGETS_SEC,
    seeds: tuple[int, ...] = SEEDS,
    out_path: Path = OUTPUT_DIR / "maxdiv_constrained.jsonl",
) -> list[RunRecord]:
    """max-div's budget series on the constrained cells, MIN_SEPARATION only.

    Defaults are the published protocol; pass smaller values only for validation runs.
    """
    records: list[RunRecord] = []
    for name in problems:
        for size in sizes:
            problem = build_problem(name, n=size, diversity_metric=CONSTRAINED_METRIC)
            records += run_maxdiv_budget_series(
                problem,
                problem_name=name,
                size=size,
                time_budgets_sec=time_budgets_sec,
                seeds=seeds,
            )
            print(f"max-div {name} size={size} done ({len(records)} records so far)", flush=True)
            save_records(records, out_path)
    return records


def main() -> None:
    """Run the full tier-2 scenario, competitors and max-div alike, and persist all records."""
    print("tier-2 competitors unconstrained ...", flush=True)
    run_competitors_unconstrained()
    print("tier-2 max-div unconstrained ...", flush=True)
    run_maxdiv_unconstrained()
    print("tier-2 competitors constrained ...", flush=True)
    run_competitors_constrained()
    print("tier-2 max-div constrained ...", flush=True)
    run_maxdiv_constrained()
    print("tier-2 complete", flush=True)


if __name__ == "__main__":
    main()
