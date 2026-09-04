"""The published tier-1 run: max-div vs. exact-solver references.

Run with: ``uv run --group benchmarks python -m benchmarks.tier1.full``.
Emits JSON/JSONL result files into ``reports/benchmarks/tier1/``; the docs artifacts are
generated separately by ``benchmarks.tier1.report``. Expect a ~6 h sequential run, most of
it spent in the two long exact reference solves.

Each experiment's exact-solver half and max-div half write separate files, so max-div can
be re-measured alone (``benchmarks.tier1.rerun``) while the exact references stay fixed.
The exact outputs (``*.json``) share their names with the tracked reference copies under
``benchmarks/tier1/data/``: promoting a fresh exact measurement means copying the files over
by hand, so a re-run never changes the tracked references by itself.

Three experiments:

1. **Max-min gap to proven optimum** — CP-SAT (threshold binary search) proves the optimum
   on U1 + C1 at n = 100/200/300; max-div runs its budget series on the same problems.
   Above n ~ 300 CP-SAT no longer proves within the cap, which bounds the experiment.
2. **Backend scaling (mean/geomean)** — how far SCIP and CP-SAT push the NN-assignment
   model before proofs stop, on a d=4 random family below the smallest generator size
   (n = 100). The experiment substantiates why no mean/geomean gap-to-optimum is published.
3. **Incumbent-at-budget geomean panel** — on shipped problems no solver can certify
   (U3 and C4 at size 1), CP-SAT runs at a generous cap and its best-found solution is
   compared against max-div's budget series. Uncertified by construction.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.common import build_problem, save_records, time_budget_series
from benchmarks.exact import solve_maxmin_cpsat, solve_nn_assignment_cpsat, solve_nn_separation_scip
from benchmarks.runners import run_maxdiv_budget_series
from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem

OUTPUT_DIR = Path("reports/benchmarks/tier1")

SEEDS = (0, 1, 2)
TIME_BUDGETS_SEC = time_budget_series(0.001, 10.0)

# Experiment 1: max-min gap to proven optimum.
MAXMIN_PROBLEMS = ("U1", "C1")
MAXMIN_SIZES = (100, 200, 300)  # CP-SAT stops proving within the cap at n ~ 400
MAXMIN_CAP_SEC = 120.0

# Experiment 2: backend scaling over increasing n (custom d=4 family; k = n // 10).
SCALING_NS = (40, 50, 60, 70, 80, 90, 100)
SCALING_DIMENSIONS = 4
SCALING_CAPS_SEC = {"SCIP (1 thread)": 900.0, "CP-SAT (1 worker)": 3600.0, "CP-SAT (8 workers)": 3600.0}

# Experiment 3: incumbent-at-budget geomean panel on shipped problems.
INCUMBENT_CASES = (("U3", 100, 10_800.0), ("C4", 150, 900.0))  # C4's bound stops improving before 900 s


def scaling_problem(n: int) -> MaxDivProblem:
    """Build the custom sub-n=100 problem family for the backend scaling experiment."""
    rng = np.random.default_rng(0)
    vectors = rng.random((n, SCALING_DIMENSIONS), dtype=np.float32)
    return MaxDivProblem.new(vectors, k=max(2, n // 10), diversity_metric=DiversityMetric.GEOMEAN_SEPARATION)


def run_maxmin_exact(out_path: Path = OUTPUT_DIR / "maxmin_exact.json") -> None:
    """Experiment 1, exact half: prove the max-min optima with CP-SAT."""
    exact_rows = []
    for name in MAXMIN_PROBLEMS:
        for size in MAXMIN_SIZES:
            problem = build_problem(name, n=size, diversity_metric=DiversityMetric.MIN_SEPARATION)
            res = solve_maxmin_cpsat(problem, time_limit_sec=MAXMIN_CAP_SEC)
            exact_rows.append(
                {
                    "problem": name,
                    "size": size,
                    "n": problem.n,
                    "k": problem.k,
                    "m": problem.m,
                    "optimum": res.min_separation,
                    "proven_optimal": res.proven_optimal,
                    "measured_sec": res.measured_sec,
                }
            )
            print(
                f"maxmin {name} size={size}: optimum={res.min_separation:.5f} proven={res.proven_optimal}", flush=True
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(exact_rows, indent=2))


def run_maxmin_maxdiv(
    problems: tuple[str, ...] = MAXMIN_PROBLEMS,
    sizes: tuple[int, ...] = MAXMIN_SIZES,
    time_budgets_sec: list[float] = TIME_BUDGETS_SEC,
    seeds: tuple[int, ...] = SEEDS,
    out_path: Path = OUTPUT_DIR / "maxmin_records.jsonl",
) -> None:
    """Experiment 1, max-div half: run max-div's budget series on the proven-optimum problems.

    Defaults are the published protocol; pass smaller values only for validation runs.
    """
    records = []
    for name in problems:
        for size in sizes:
            problem = build_problem(name, n=size, diversity_metric=DiversityMetric.MIN_SEPARATION)
            records += run_maxdiv_budget_series(
                problem, problem_name=name, size=size, time_budgets_sec=time_budgets_sec, seeds=seeds
            )
            print(f"maxmin max-div {name} size={size} done", flush=True)
    save_records(records, out_path)


def run_backend_scaling() -> None:
    """Experiment 2: run each backend over increasing n until its first failed proof."""
    rows = []
    for backend, cap in SCALING_CAPS_SEC.items():
        for n in SCALING_NS:
            problem = scaling_problem(n)
            t0 = time.perf_counter()
            if backend.startswith("SCIP"):
                try:
                    res = solve_nn_separation_scip(problem, DiversityMetric.GEOMEAN_SEPARATION, time_limit_sec=cap)
                    proven, measured = res.proven_optimal, res.measured_sec
                except RuntimeError:
                    proven, measured = False, time.perf_counter() - t0
            else:
                workers = 1 if "1 worker" in backend else 8
                try:
                    res = solve_nn_assignment_cpsat(
                        problem, DiversityMetric.GEOMEAN_SEPARATION, time_limit_sec=cap, num_workers=workers
                    )
                    proven, measured = res.proven_optimal, res.measured_sec
                except RuntimeError:
                    proven, measured = False, time.perf_counter() - t0
            rows.append({"backend": backend, "n": n, "k": problem.k, "measured_sec": measured, "proven": proven})
            print(f"scaling {backend} n={n}: proven={proven} in {measured:.1f}s", flush=True)
            if not proven:
                break  # larger n would only time out again, at the full cap cost
    (OUTPUT_DIR / "scaling.json").write_text(json.dumps(rows, indent=2))


def run_incumbent_exact(out_path: Path = OUTPUT_DIR / "incumbent.json") -> None:
    """Experiment 3, exact half: long-cap CP-SAT incumbents on the uncertifiable problems."""
    panel_rows = []
    for name, size, cap in INCUMBENT_CASES:
        problem = build_problem(name, n=size, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION)
        res = solve_nn_assignment_cpsat(problem, DiversityMetric.GEOMEAN_SEPARATION, time_limit_sec=cap)
        panel_rows.append(
            {
                "problem": name,
                "size": size,
                "n": problem.n,
                "k": problem.k,
                "m": problem.m,
                "cap_sec": cap,
                **asdict(res) | {"i_selected": res.i_selected.tolist()},
            }
        )
        print(
            f"incumbent {name} size={size}: value={res.objective_value:.5f} bound={res.objective_bound:.5f} "
            f"proven={res.proven_optimal} in {res.measured_sec:.0f}s",
            flush=True,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(panel_rows, indent=2))


def run_incumbent_maxdiv(
    cases: tuple[tuple[str, int, float], ...] = INCUMBENT_CASES,
    time_budgets_sec: list[float] = TIME_BUDGETS_SEC,
    seeds: tuple[int, ...] = SEEDS,
    out_path: Path = OUTPUT_DIR / "incumbent_records.jsonl",
) -> None:
    """Experiment 3, max-div half: run max-div's budget series on the incumbent-panel problems.

    Defaults are the published protocol; pass smaller values only for validation runs.
    """
    records = []
    for name, size, _cap in cases:
        problem = build_problem(name, n=size, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION)
        records += run_maxdiv_budget_series(
            problem, problem_name=name, size=size, time_budgets_sec=time_budgets_sec, seeds=seeds
        )
        print(f"incumbent max-div {name} size={size} done", flush=True)
    save_records(records, out_path)


def main() -> None:
    """Run all three tier-1 experiments, exact and max-div halves alike."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("tier-1 experiment 1: max-min gap to proven optimum ...", flush=True)
    run_maxmin_exact()
    run_maxmin_maxdiv()
    print("tier-1 experiment 2: backend scaling ...", flush=True)
    run_backend_scaling()
    print("tier-1 experiment 3: incumbent-at-budget panel ...", flush=True)
    run_incumbent_exact()
    run_incumbent_maxdiv()
    print("tier-1 complete", flush=True)


if __name__ == "__main__":
    main()
