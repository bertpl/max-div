"""Certified feasibility verdicts for the constrained benchmark problems.

For each constrained benchmark problem and each size in `VERDICT_SIZES`, this runs the feasibility
pipeline at its maximum construction budget, `CONSTRUCTION_MAX_ITER`, and reports the verdict, plus
the constraints-score ceiling wherever infeasibility is certified. The output feeds the committed
per-problem verdict tables in the docs.
"""

import numpy as np

from max_div._core._markdown import Report, Table
from max_div._core._utils import stdout_to_file
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.constraints import ConstraintList
from max_div._core.feasibility import CONSTRUCTION_MAX_ITER, FeasibilityResult, FeasibilityStatus, find_feasible
from max_div._core.metrics import DiversityMetric
from max_div._core.problem import VectorMaxDivProblem
from max_div._core.solver._score import ScoreGenerator

# The verdict tables sweep a 1-2-5 size series reaching well past the strategy benchmarks' grid,
# far enough that constrained problems whose difficulty compounds with size cross into certified
# infeasibility inside the table rather than beyond its last row.
VERDICT_SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Verdicts are properties of the problem, not measurements, but they still depend on the search
# budget (a bigger budget can only move UNKNOWN toward a proof) and on the seed of the candidate
# noise. Both are pinned so regenerating the committed tables is deterministic.
VERDICT_SEED = 0
TURBO_MAX_ITER = 200  # reduced ascent budget for --turbo smoke runs
TURBO_N_SIZES = 3  # number of ladder sizes retained for --turbo smoke runs


def _constrained_problem_names() -> list[str]:
    """Return the registered benchmark problems that carry constraints, by name."""
    names = BenchmarkProblemFactory.get_all_benchmark_names()
    smallest = min(VERDICT_SIZES)
    return [name for name in names if BenchmarkProblemFactory.get_problem_dimensions(name, smallest)[3] > 0]


def _verdict_for(problem: VectorMaxDivProblem, max_iter: int) -> FeasibilityResult:
    """Run the feasibility pipeline on one problem instance at the pinned seed."""
    con_values, con_indices = ConstraintList(problem.constraints).to_numpy()
    return find_feasible(
        con_values=con_values,
        con_indices=con_indices,
        con_weights=np.array([con.weight for con in problem.constraints], dtype=np.float64),
        n=problem.n,
        k=problem.k,
        max_iter=max_iter,
        seed=VERDICT_SEED,
        stop_at_first_proof=False,
    )


def _verdict_cell(status: FeasibilityStatus) -> str:
    """Return the verdict column cell: the status name, colored when it is a proof."""
    if status == FeasibilityStatus.FEASIBLE:
        return f'<span style="color:{Table.GREEN}">**feasible** (proven)</span>'
    if status == FeasibilityStatus.INFEASIBLE:
        return f'<span style="color:{Table.RED}">**infeasible** (proven)</span>'
    return "unknown"


def _ceiling_cell(problem: VectorMaxDivProblem, result: FeasibilityResult) -> str:
    """Return the constraints-score-ceiling cell for one verdict row.

    A witness attains the perfect score, so feasible rows state 1.0 exactly; UNKNOWN certifies
    nothing, so the cell shows a dash.
    """
    if result.status == FeasibilityStatus.FEASIBLE:
        return "1.0"
    if result.status == FeasibilityStatus.UNKNOWN:
        return "-"
    score_generator = ScoreGenerator(
        n=problem.n,
        k=problem.k,
        diversity_metric=problem.diversity_metric,
        diversity_tie_breakers=[],
        constraints=problem.constraints,
    )
    return f"{score_generator.constraints_score_for_violation(result.violation_floor):.5f}"


def _build_report(problem_name: str, sizes: list[int], max_iter: int) -> Report:
    """Build the verdict table report for one constrained problem."""
    report = Report()
    report += (
        f"Certified feasibility verdicts for problem {problem_name} "
        f"(ascent budget {max_iter} iterations, thorough mode, fixed seed):"
    )
    table = Table(["$n$", "$d$", "$k$", "$m$", "Verdict", "Constraints-score ceiling"])
    for n in sizes:
        d, _, k, m, _ = BenchmarkProblemFactory.get_problem_dimensions(problem_name, n)
        problem = BenchmarkProblemFactory.construct_problem(
            name=problem_name, n=n, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION
        )
        result = _verdict_for(problem, max_iter)
        table.add_row([str(n), str(d), str(k), str(m), _verdict_cell(result.status), _ceiling_cell(problem, result)])
    report += table
    return report


def run_solver_feasibility_benchmark(name: str, markdown: bool, file: bool, turbo: bool) -> None:
    """Generate per-size feasibility verdict tables for the constrained benchmark problems.

    Args:
        name: A constrained benchmark problem name, or `"all"` for every constrained problem.
        markdown: Render tables as markdown rather than console layout.
        file: Write each problem's report to `feasibility_verdicts_<name>.md` instead of stdout.
        turbo: Shrink the size grid and ascent budget for a fast smoke run; the resulting
            verdicts are weaker (more UNKNOWN) and not meant for the docs.
    """
    constrained = _constrained_problem_names()
    problem_names = constrained if name == "all" else [name]
    unknown = [p for p in problem_names if p not in constrained]
    if unknown:
        raise ValueError(f"Not a constrained benchmark problem: {', '.join(unknown)}")

    sizes = VERDICT_SIZES[:TURBO_N_SIZES] if turbo else VERDICT_SIZES
    max_iter = TURBO_MAX_ITER if turbo else CONSTRUCTION_MAX_ITER

    for problem_name in problem_names:
        report = _build_report(problem_name, sizes, max_iter)
        with stdout_to_file(enabled=file, filename=f"feasibility_verdicts_{problem_name}.md"):
            report.print(markdown=markdown or file)
        if file:
            print(f"Wrote feasibility_verdicts_{problem_name}.md")
