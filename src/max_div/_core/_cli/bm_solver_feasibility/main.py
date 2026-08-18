"""Certified feasibility verdicts for the constrained benchmark problems.

For each constrained benchmark problem and each size on the shared k-ladder, this runs
`check_feasibility` at the pipeline's verdict-grade budget, `FEASIBILITY_MAX_ITER_HIGH`, and
reports the verdict, plus the constraints-score ceiling wherever infeasibility is certified. The
output feeds the committed per-problem verdict tables in the docs.
"""

from max_div._core._cli.bm_solver_sizing import K_LADDER, determine_problem_size_for_k
from max_div._core._markdown import Report, Table
from max_div._core._utils import stdout_to_file
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.feasibility import (
    FEASIBILITY_MAX_ITER_HIGH,
    FEASIBILITY_MAX_ITER_LOW,
    FeasibilityResult,
    FeasibilityStatus,
)
from max_div._core.metrics import DiversityMetric
from max_div._core.problem import VectorMaxDivProblem
from max_div._core.solver._score import ScoreGenerator

# Verdicts are properties of the problem, not measurements, but they still depend on the search
# budget (a bigger budget can only move UNKNOWN toward a proof) and on the seed of the candidate
# noise. The budget is pinned here and check_feasibility pins the seed, so regenerating the
# committed tables is deterministic.
TURBO_N_SIZES = 3  # number of ladder rungs retained for --turbo smoke runs


def _constrained_problem_names() -> list[str]:
    """Return the registered benchmark problems that carry constraints, by name."""
    names = BenchmarkProblemFactory.get_all_benchmark_names()
    return [name for name in names if _has_constraints(name)]


def _has_constraints(problem_name: str) -> bool:
    """Report whether the problem carries constraints, a size-independent property probed at the smallest ladder k."""
    probe_n = determine_problem_size_for_k(problem_name, K_LADDER[0])
    return BenchmarkProblemFactory.get_problem_dimensions(problem_name, probe_n)[3] > 0


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
        f"Certified feasibility verdicts for problem {problem_name}, "
        f"computed with `check_feasibility(thorough=True, max_iter={max_iter})`:"
    )
    table = Table(["$n$", "$d$", "$k$", "$m$", "Verdict", "Constraints-score ceiling"])
    for n in sizes:
        d, _, k, m, _ = BenchmarkProblemFactory.get_problem_dimensions(problem_name, n)
        problem = BenchmarkProblemFactory.construct_problem(
            name=problem_name, n=n, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION
        )
        result = problem.check_feasibility(thorough=True, max_iter=max_iter)
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

    ladder = K_LADDER[:TURBO_N_SIZES] if turbo else K_LADDER
    max_iter = FEASIBILITY_MAX_ITER_LOW if turbo else FEASIBILITY_MAX_ITER_HIGH

    for problem_name in problem_names:
        sizes = [determine_problem_size_for_k(problem_name, k) for k in ladder]
        report = _build_report(problem_name, sizes, max_iter)
        with stdout_to_file(enabled=file, filename=f"feasibility_verdicts_{problem_name}.md"):
            report.print(markdown=markdown or file)
        if file:
            print(f"Wrote feasibility_verdicts_{problem_name}.md")
