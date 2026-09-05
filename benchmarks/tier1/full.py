"""Run the tier-1 comparison: max-div against the exact solvers' certified optima.

Run with: ``uv run --group benchmarks python -m benchmarks.tier1.full``.
Writes the exact solvers' results as JSON and max-div's records as JSONL into `OUTPUT_DIR`; the
docs artifacts come from ``benchmarks.tier1.report``.

Two halves, so max-div can be re-measured alone (``benchmarks.tier1.rerun``) while the exact
references stay fixed:

- **Exact half.** For U1 and C1, every solver ascends the 1-2-5 size grid from 20 and runs each
  size to proven optimality or the certification cap; a solver stops at its first size not
  certified. Max-min runs on CP-SAT (threshold search), SCIP and HiGHS (big-M MIP); mean- and
  geomean-of-NN separation run on CP-SAT's nearest-neighbor assignment model alone, the only
  backend that certifies that model beyond the smallest sizes. The JSON outputs share their names
  with the tracked reference copies under `DATA_DIR`: promoting a fresh exact measurement means
  copying the files over by hand.
- **max-div half.** On every (problem, objective, size) cell some solver certified, both budget
  series (see ``benchmarks.common.protocol``), one independent solve per budget and seed.

Both halves skip cells already on file, so an interrupted run resumes by rerunning.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.common import build_problem, load_records, save_records
from benchmarks.common.protocol import (
    CERTIFICATION_CAP_SEC,
    MULTI_WORKER_BUDGETS_SEC,
    N_WORKERS,
    SEEDS,
    SINGLE_WORKER_BUDGETS_SEC,
    SINGLE_WORKER_CONCURRENCY,
)
from benchmarks.common.records import RunRecord
from benchmarks.exact import solve_maxmin_cpsat, solve_maxmin_highs, solve_maxmin_scip, solve_nn_assignment_cpsat
from benchmarks.runners import run_maxdiv_budget_series
from benchmarks.runners.maxdiv_runner import maxdiv_tool_label
from benchmarks.solver_scaling.grid import size_grid
from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem

OUTPUT_DIR = Path("reports/benchmarks/tier1")
DATA_DIR = Path(__file__).parent / "data"
EXACT_MAXMIN_FILE = "exact_maxmin.json"
EXACT_NN_FILE = "exact_nn.json"

PROBLEMS = ("U1", "C1")
# The grid is walked until a solver stops certifying; this bound only guards against a solver that never stops.
GRID_BOUND = 5000
NN_OBJECTIVES = (DiversityMetric.MEAN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION)


@dataclass(frozen=True)
class CertifiedOptimum:
    """Record what one exact solve found: the objective value, whether it is proven optimal, and how long it took."""

    optimum: float | None
    proven_optimal: bool
    measured_sec: float


ExactSolve = Callable[[MaxDivProblem], CertifiedOptimum]


def _cpsat_maxmin(problem: MaxDivProblem) -> CertifiedOptimum:
    """Certify the max-min optimum with CP-SAT's threshold search, on the protocol's worker count."""
    result = solve_maxmin_cpsat(problem, time_limit_sec=CERTIFICATION_CAP_SEC, num_workers=N_WORKERS)
    return CertifiedOptimum(result.min_separation, result.proven_optimal, result.measured_sec)


def _scip_maxmin(problem: MaxDivProblem) -> CertifiedOptimum:
    """Certify the max-min optimum with SCIP's big-M MIP (single-threaded: the PyPI build has no concurrent solve)."""
    result = solve_maxmin_scip(problem, time_limit_sec=CERTIFICATION_CAP_SEC)
    return CertifiedOptimum(result.min_separation, result.proven_optimal, result.measured_sec)


def _highs_maxmin(problem: MaxDivProblem) -> CertifiedOptimum:
    """Certify the max-min optimum with HiGHS's big-M MIP on the protocol's worker count."""
    result = solve_maxmin_highs(problem, time_limit_sec=CERTIFICATION_CAP_SEC, num_workers=N_WORKERS)
    return CertifiedOptimum(result.min_separation, result.proven_optimal, result.measured_sec)


# Max-min solvers by registry key.
MAXMIN_SOLVERS: dict[str, ExactSolve] = {"ortools-cpsat": _cpsat_maxmin, "scip": _scip_maxmin, "highs": _highs_maxmin}


def _cpsat_nn(metric: DiversityMetric) -> ExactSolve:
    """Build the CP-SAT nearest-neighbor assignment certifier for one NN objective."""

    def solve(problem: MaxDivProblem) -> CertifiedOptimum:
        result = solve_nn_assignment_cpsat(problem, metric, time_limit_sec=CERTIFICATION_CAP_SEC, num_workers=N_WORKERS)
        return CertifiedOptimum(result.objective_value, result.proven_optimal, result.measured_sec)

    return solve


def _load_rows(path: Path) -> list[dict]:
    """Read the exact rows on file, or none."""
    return json.loads(path.read_text()) if path.exists() else []


def _certify_increasing_sizes(
    rows: list[dict], path: Path, problem_name: str, metric: DiversityMetric, solver_key: str, solve: ExactSolve
) -> None:
    """Run one solver up the grid on one (problem, objective) until its first size not certified.

    Every attempt is recorded, the failed one included: the results page states where each
    solver's certification stopped. Rows already on file are skipped, and the ascent resumes
    after them.
    """
    done = {(r["problem"], r["objective"], r["solver"], r["n"]): r for r in rows}
    for n in size_grid(GRID_BOUND):
        row = done.get((problem_name, metric.name, solver_key, n))
        if row is None:
            problem = build_problem(problem_name, n=n, diversity_metric=metric)
            try:
                certified = solve(problem)
            except RuntimeError as error:  # no solution at all within the cap
                certified = CertifiedOptimum(None, False, CERTIFICATION_CAP_SEC)
                print(f"  {solver_key} {problem_name} {metric.name} n={n}: no solution ({error})", flush=True)
            row = {
                "problem": problem_name,
                "objective": metric.name,
                "solver": solver_key,
                "n": n,
                "k": problem.k,
                "m": problem.m,
                **asdict(certified),
            }
            rows.append(row)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, indent=2))
        print(
            f"  {solver_key} {problem_name} {metric.name} n={n}: "
            f"{'certified' if row['proven_optimal'] else 'not certified'} in {row['measured_sec']:.1f} s",
            flush=True,
        )
        if not row["proven_optimal"]:
            return


def run_exact_maxmin(out_path: Path = OUTPUT_DIR / EXACT_MAXMIN_FILE) -> list[dict]:
    """Run the exact half for max-min: every solver ascends the grid on U1 and C1 until it stops certifying."""
    rows = _load_rows(out_path)
    for problem_name in PROBLEMS:
        for solver_key, solve in MAXMIN_SOLVERS.items():
            _certify_increasing_sizes(rows, out_path, problem_name, DiversityMetric.MIN_SEPARATION, solver_key, solve)
    return rows


def run_exact_nn(out_path: Path = OUTPUT_DIR / EXACT_NN_FILE) -> list[dict]:
    """Run the exact half for mean/geomean of NN: CP-SAT ascends the grid on U1 and C1 until it stops certifying."""
    rows = _load_rows(out_path)
    for problem_name in PROBLEMS:
        for metric in NN_OBJECTIVES:
            _certify_increasing_sizes(rows, out_path, problem_name, metric, "ortools-cpsat", _cpsat_nn(metric))
    return rows


def certified_sizes(exact_rows: list[dict]) -> dict[tuple[str, str], list[int]]:
    """Return, per (problem, objective), the sizes at which at least one solver certified the optimum."""
    sizes: dict[tuple[str, str], set[int]] = {}
    for row in exact_rows:
        if row["proven_optimal"]:
            sizes.setdefault((row["problem"], row["objective"]), set()).add(row["n"])
    return {key: sorted(values) for key, values in sizes.items()}


def maxdiv_records_path(metric: DiversityMetric, records_dir: Path = OUTPUT_DIR) -> Path:
    """Return the JSONL file holding max-div's records for one objective."""
    return records_dir / f"maxdiv_{metric.name.lower()}.jsonl"


def run_maxdiv(
    exact_rows: list[dict],
    seeds: tuple[int, ...] = SEEDS,
    single_budgets_sec: list[float] = SINGLE_WORKER_BUDGETS_SEC,
    multi_budgets_sec: list[float] = MULTI_WORKER_BUDGETS_SEC,
    n_workers: int = N_WORKERS,
    records_dir: Path = OUTPUT_DIR,
) -> None:
    """Run the max-div half: both budget series on every certified (problem, objective, size) cell.

    Defaults are the published protocol; pass smaller values only for validation runs. Cells
    whose records are already on file are skipped.
    """
    for (problem_name, objective), sizes in sorted(certified_sizes(exact_rows).items()):
        metric = DiversityMetric[objective]
        path = maxdiv_records_path(metric, records_dir)
        records: list[RunRecord] = load_records(path) if path.exists() else []
        done = {(r.problem, r.n, r.tool) for r in records}
        for n in sizes:
            problem = build_problem(problem_name, n=n, diversity_metric=metric)
            for budgets, workers in ((single_budgets_sec, 1), (multi_budgets_sec, n_workers)):
                if (problem_name, n, maxdiv_tool_label(n_workers=workers)) in done:
                    continue
                records += run_maxdiv_budget_series(
                    problem,
                    problem_name=problem_name,
                    size=n,
                    time_budgets_sec=budgets,
                    seeds=seeds,
                    n_workers=workers,
                    concurrency=SINGLE_WORKER_CONCURRENCY if workers == 1 else 1,
                )
                save_records(records, path)
            print(f"max-div {problem_name} {objective} n={n} done", flush=True)


def main() -> None:
    """Run both halves: the exact references, then max-div on every certified cell."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("tier-1 exact half: max-min ...", flush=True)
    maxmin_rows = run_exact_maxmin()
    print("tier-1 exact half: mean/geomean of NN ...", flush=True)
    nn_rows = run_exact_nn()
    print("tier-1 max-div half ...", flush=True)
    run_maxdiv(maxmin_rows + nn_rows)
    print("tier-1 complete", flush=True)


if __name__ == "__main__":
    main()
