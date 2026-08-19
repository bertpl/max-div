"""Run max-div across a budget series, one independent solve per rung x seed."""

from benchmarks.common.quality import evaluate_selection, n_constraints_satisfied
from benchmarks.common.records import RunRecord
from max_div.problem import MaxDivProblem
from max_div.solver import MaxDivSolverBuilder, SolverPreset, TargetDuration, Verbosity, iterations, seconds


def run_maxdiv_budget_series(
    problem: MaxDivProblem,
    problem_name: str,
    size: int,
    time_budgets_sec: list[float] | None = None,
    iteration_budgets: list[int] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    preset: SolverPreset = SolverPreset.DEFAULT,
) -> list[RunRecord]:
    """Solve the problem once per (budget, seed) and record measured time + quality.

    Args:
        problem: Problem to solve.
        problem_name: Generator name recorded in each record (e.g. ``"U1"``).
        size: Generator size parameter, recorded in each record.
        time_budgets_sec: Wall-clock ladder in seconds (may be combined with iteration budgets).
        iteration_budgets: Iteration-count ladder (recorded with an ``iterations:`` budget tag).
        seeds: One independent solve per seed per rung.
        preset: Solver preset to run.

    Returns:
        One record per (budget, seed), with measured wall-clock as reported by the solver.
    """
    budgets: list[tuple[str, TargetDuration]] = []
    for t in time_budgets_sec or []:
        budgets.append((f"time:{t}s", seconds(t)))
    for i in iteration_budgets or []:
        budgets.append((f"iterations:{i}", iterations(i)))

    records = []
    for budget_tag, target in budgets:
        for seed in seeds:
            solver = MaxDivSolverBuilder(problem).with_preset(target, preset).with_seed(seed).build()
            solution = solver.solve(verbosity=Verbosity.SILENT)
            elapsed = solution.duration
            records.append(
                RunRecord(
                    tool=f"max-div[{preset.name}]",
                    problem=problem_name,
                    size=size,
                    n=problem.n,
                    k=problem.k,
                    diversity_metric=problem.diversity_metric.name,
                    seed=seed,
                    budget=budget_tag,
                    measured_sec=elapsed.t_elapsed_sec,
                    n_iterations=elapsed.n_iterations,
                    quality=evaluate_selection(problem, solution.i_selected),
                    n_constraints=problem.m,
                    n_constraints_satisfied=n_constraints_satisfied(problem, solution.i_selected),
                )
            )
    return records
