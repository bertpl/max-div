"""Run the tier-2 comparison: max-div against the one-shot tools, on U1, min separation.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.full``.
Emits one JSONL record file per half into `OUTPUT_DIR` — entrant and max-div records are kept
apart so max-div can be re-measured alone (``benchmarks.tier2.rerun``) while the entrant half stays
fixed. Figures and tables come from ``benchmarks.tier2.report``. The tracked reference copy of the
entrant output lives under `DATA_DIR`; `benchmarks/README.md` says how it is refreshed.

One problem, `PROBLEM`, at the sizes in `SIZES`; one objective, `METRIC`. Entrants are the non-exact
registry tools; each tool runs at every size up to the largest n it finished within the
solver-scaling time budget (read from the scaling time stage's tracked records), once per seed.
max-div runs both budget series (see ``benchmarks.common.protocol``), one independent solve per
budget and seed.
"""

from pathlib import Path

from benchmarks.adapters import (
    ApricotFacilityLocation,
    CodeFdmSingleColor,
    DppyKDpp,
    FpsampleFPS,
    KMedoidsFasterPAM,
    QcSelectorMaxMin,
    QcSelectorMaxSum,
    RandomBaseline,
    RdkitMaxMin,
    SelectionAdapter,
    SkmatterFPS,
)
from benchmarks.common import build_problem, load_records, save_records
from benchmarks.common.protocol import (
    MULTI_WORKER_BUDGETS_SEC,
    N_WORKERS,
    SEEDS,
    SINGLE_WORKER_BUDGETS_SEC,
)
from benchmarks.common.records import RunRecord
from benchmarks.figures.style import tool_key
from benchmarks.runners import run_adapter, run_maxdiv_budget_series
from benchmarks.runners.maxdiv_runner import maxdiv_tool_label
from benchmarks.solver_scaling.quality_stage import time_limits
from max_div.metrics import DiversityMetric

OUTPUT_DIR = Path("reports/benchmarks/tier2")
DATA_DIR = Path(__file__).parent / "data"
ENTRANT_FILE = "third_party_u1.jsonl"
MAXDIV_FILE = "maxdiv_u1.jsonl"

PROBLEM = "U1"
SIZES = (200, 1000, 5000, 20000, 100000)
METRIC = DiversityMetric.MIN_SEPARATION


def entrant_adapters() -> list[SelectionAdapter]:
    """Return the tier-2 roster: every non-exact registry tool that takes vector input, plus the random baseline."""
    return [
        RandomBaseline(),
        FpsampleFPS(),
        FpsampleFPS(variant="kdline"),
        SkmatterFPS(),
        RdkitMaxMin(),
        ApricotFacilityLocation(),
        QcSelectorMaxMin(),
        QcSelectorMaxSum(),
        KMedoidsFasterPAM(),
        DppyKDpp(),
        CodeFdmSingleColor(),
    ]


def runs_at_size(adapter: SelectionAdapter, n: int, limits: dict[tuple[str, str], int]) -> bool:
    """Return whether a tool runs at size n: its best configuration's scaling time limit covers n.

    The random baseline has no scaling configuration and runs at every size.
    """
    key = tool_key(adapter.name)
    if key == "random":
        return True
    limit = max((n_max for (tool, _config), n_max in limits.items() if tool == key), default=0)
    return limit >= n


def run_entrants(
    sizes: tuple[int, ...] = SIZES,
    seeds: tuple[int, ...] = SEEDS,
    adapters: list[SelectionAdapter] | None = None,
    limits: dict[tuple[str, str], int] | None = None,
    out_path: Path = OUTPUT_DIR / ENTRANT_FILE,
) -> list[RunRecord]:
    """Run the entrant half: every entrant once per seed at every size it runs at."""
    adapters = entrant_adapters() if adapters is None else adapters
    limits = time_limits() if limits is None else limits
    records: list[RunRecord] = load_records(out_path) if out_path.exists() else []
    done = {(r.n, r.tool) for r in records}
    for n in sizes:
        problem = build_problem(PROBLEM, n=n, diversity_metric=METRIC)
        for adapter in adapters:
            if (n, adapter.name) in done or not runs_at_size(adapter, n, limits):
                continue
            records += run_adapter(adapter, problem, problem_name=PROBLEM, size=n, seeds=seeds)
            save_records(records, out_path)
        print(f"entrants {PROBLEM} n={n} done ({len(records)} records so far)", flush=True)
    return records


def run_maxdiv(
    sizes: tuple[int, ...] = SIZES,
    seeds: tuple[int, ...] = SEEDS,
    single_budgets_sec: list[float] = SINGLE_WORKER_BUDGETS_SEC,
    multi_budgets_sec: list[float] = MULTI_WORKER_BUDGETS_SEC,
    n_workers: int = N_WORKERS,
    out_path: Path = OUTPUT_DIR / MAXDIV_FILE,
) -> list[RunRecord]:
    """Run the max-div half: both budget series at every size, one solve at a time.

    The single-worker series is not packed across processes here: from n = 20,000 the distance
    computation of twelve side-by-side solves contends for the cores and inflates every measured
    time by close to a second, and the whole series costs only minutes per size unpacked.
    Defaults are the published protocol; pass smaller values only for validation runs.
    """
    records: list[RunRecord] = load_records(out_path) if out_path.exists() else []
    done = {(r.n, r.tool) for r in records}
    for n in sizes:
        problem = build_problem(PROBLEM, n=n, diversity_metric=METRIC)
        for budgets, workers in ((single_budgets_sec, 1), (multi_budgets_sec, n_workers)):
            if (n, maxdiv_tool_label(n_workers=workers)) in done:
                continue
            records += run_maxdiv_budget_series(
                problem,
                problem_name=PROBLEM,
                size=n,
                time_budgets_sec=budgets,
                seeds=seeds,
                n_workers=workers,
            )
            save_records(records, out_path)
        print(f"max-div {PROBLEM} n={n} done ({len(records)} records so far)", flush=True)
    return records


def main() -> None:
    """Run both halves and persist all records."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("tier-2 entrants ...", flush=True)
    run_entrants()
    print("tier-2 max-div ...", flush=True)
    run_maxdiv()
    print("tier-2 complete", flush=True)


if __name__ == "__main__":
    main()
