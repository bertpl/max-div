"""Run the tier-3 comparison: max-div and the one-shot tools on the MDPLIB MMDP instances.

Run with: ``uv run --group benchmarks python -m benchmarks.tier3.full``.
Emits two JSONL record files into ``reports/benchmarks/tier3/`` — the entrants' records and
max-div's — so max-div can be re-measured alone while the entrant side stays fixed. The entrant
output shares its name with the tracked reference copy under ``benchmarks/tier3/data/``:
promoting a fresh entrant measurement means copying the file over by hand. The docs artifacts
come from ``benchmarks.tier3.report``.

Protocol (mirrored on the results page):

- Every published (instance, k) pairing of the Geo and Ran sets runs max-div's single-worker
  budget series; the n = 500 pairings also run the multi-worker series, the smaller ones plateau
  within milliseconds. The Glover pairings (n <= 30) run a short single-worker series only: they
  are reported as a match count, not charted.
- Entrants are the non-exact registry tools whose input the instance family provides: Geo carries
  coordinates, so every tool enters; Ran carries a distance matrix, so only the tools that accept
  one enter. One run per seed. Glover has no entrants.

Instances are fetched from the MDPLIB site at run time and never redistributed. Both halves skip
cells already on file, so an interrupted run resumes by rerunning.
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
    RdkitMaxMin,
    SelectionAdapter,
    SkmatterFPS,
)
from benchmarks.common import grid_budget_series, load_records, save_records
from benchmarks.common.protocol import MULTI_WORKER_BUDGETS_SEC, N_WORKERS, SEEDS, SINGLE_WORKER_BUDGETS_SEC
from benchmarks.common.records import RunRecord
from benchmarks.mdplib import load_instance
from benchmarks.mdplib.best_known import BestKnown, load_best_known
from benchmarks.runners import run_adapter, run_maxdiv_budget_series
from benchmarks.runners.maxdiv_runner import maxdiv_tool_label
from max_div.metrics import DiversityMetric

OUTPUT_DIR = Path("reports/benchmarks/tier3")
DATA_DIR = Path(__file__).parent / "data"
ENTRANT_FILE = "third_party_mdplib.jsonl"
MAXDIV_FILE = "maxdiv_mdplib.jsonl"

METRIC = DiversityMetric.MIN_SEPARATION  # the published MMDP values are max-min
CHARTED_FAMILIES = ("Geo", "Ran")
MULTI_WORKER_MIN_N = 500
GLOVER_BUDGETS_SEC = grid_budget_series(0.001, 1.0)
SINGLE_WORKER_CONCURRENCY = N_WORKERS


def entrant_adapters(family: str) -> list[SelectionAdapter]:
    """The entrants of one instance family: every tool whose input form the family provides."""
    distance_matrix_tools: list[SelectionAdapter] = [QcSelectorMaxMin(), QcSelectorMaxSum(), KMedoidsFasterPAM()]
    if family == "Ran":
        return distance_matrix_tools
    if family == "Geo":
        vector_tools: list[SelectionAdapter] = [
            FpsampleFPS(),
            FpsampleFPS(variant="kdline"),
            SkmatterFPS(),
            RdkitMaxMin(),
            ApricotFacilityLocation(),
            DppyKDpp(),
            CodeFdmSingleColor(),
        ]
        return vector_tools + distance_matrix_tools
    return []


def run_entrants(
    pairings: list[BestKnown] | None = None,
    seeds: tuple[int, ...] = SEEDS,
    out_path: Path = OUTPUT_DIR / ENTRANT_FILE,
) -> list[RunRecord]:
    """Entrant half: every entrant once per seed on every charted pairing."""
    pairings = load_best_known() if pairings is None else pairings
    records: list[RunRecord] = load_records(out_path) if out_path.exists() else []
    done = {(r.problem, r.size, r.tool) for r in records}
    for row in pairings:
        if row.family not in CHARTED_FAMILIES:
            continue
        adapters = [a for a in entrant_adapters(row.family) if (row.instance, row.k, a.name) not in done]
        if not adapters:
            continue
        problem = load_instance(row.family, row.instance, k=row.k, diversity_metric=METRIC)
        for adapter in adapters:
            records += run_adapter(adapter, problem, problem_name=row.instance, size=row.k, seeds=seeds)
        save_records(records, out_path)
        print(f"entrants {row.instance} k={row.k} done ({len(records)} records so far)", flush=True)
    return records


def run_maxdiv(
    pairings: list[BestKnown] | None = None,
    seeds: tuple[int, ...] = SEEDS,
    single_budgets_sec: list[float] = SINGLE_WORKER_BUDGETS_SEC,
    multi_budgets_sec: list[float] = MULTI_WORKER_BUDGETS_SEC,
    glover_budgets_sec: list[float] = GLOVER_BUDGETS_SEC,
    n_workers: int = N_WORKERS,
    out_path: Path = OUTPUT_DIR / MAXDIV_FILE,
) -> list[RunRecord]:
    """max-div half: the single-worker series on every pairing, the multi-worker series on the largest instances.

    Defaults are the published protocol; pass smaller values only for validation runs. `size`
    carries k in every record: the instance file name plus k identifies a published pairing.
    """
    pairings = load_best_known() if pairings is None else pairings
    records: list[RunRecord] = load_records(out_path) if out_path.exists() else []
    done = {(r.problem, r.size, r.tool) for r in records}
    for row in pairings:
        series: list[tuple[list[float], int]] = []
        if row.family in CHARTED_FAMILIES:
            series.append((single_budgets_sec, 1))
            if row.n >= MULTI_WORKER_MIN_N:
                series.append((multi_budgets_sec, n_workers))
        else:
            series.append((glover_budgets_sec, 1))
        series = [(budgets, workers) for budgets, workers in series if (row.instance, row.k, maxdiv_tool_label(n_workers=workers)) not in done]
        if not series:
            continue
        problem = load_instance(row.family, row.instance, k=row.k, diversity_metric=METRIC)
        for budgets, workers in series:
            records += run_maxdiv_budget_series(
                problem,
                problem_name=row.instance,
                size=row.k,
                time_budgets_sec=budgets,
                seeds=seeds,
                n_workers=workers,
                concurrency=SINGLE_WORKER_CONCURRENCY if workers == 1 else 1,
            )
        save_records(records, out_path)
        print(f"max-div {row.instance} k={row.k} done ({len(records)} records so far)", flush=True)
    return records


def main() -> None:
    """Run both halves and persist all records."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("tier-3 entrants ...", flush=True)
    run_entrants()
    print("tier-3 max-div ...", flush=True)
    run_maxdiv()
    print("tier-3 complete", flush=True)


if __name__ == "__main__":
    main()
