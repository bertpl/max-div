"""Run max-div across a budget series, one independent solve per budget x seed, with one or several workers."""

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from benchmarks.common.quality import evaluate_selection, n_constraints_satisfied
from benchmarks.common.records import RunRecord
from max_div.problem import MaxDivProblem
from max_div.solver import (
    MaxDivSolverBuilder,
    ParallelMaxDivSolverBuilder,
    SolverPreset,
    TargetDuration,
    Verbosity,
    iterations,
    seconds,
)


def maxdiv_tool_label(preset: SolverPreset = SolverPreset.DEFAULT, n_workers: int = 1) -> str:
    """Return the record label of a max-div series: the preset, plus the worker count when several."""
    return f"max-div[{preset.name}]" if n_workers == 1 else f"max-div[{preset.name}, {n_workers} workers]"


def run_maxdiv_budget_series(
    problem: MaxDivProblem,
    problem_name: str,
    size: int,
    time_budgets_sec: list[float] | None = None,
    iteration_budgets: list[int] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    preset: SolverPreset = SolverPreset.DEFAULT,
    n_workers: int = 1,
    concurrency: int = 1,
) -> list[RunRecord]:
    """Solve the problem once per (budget, seed) and record measured time + quality.

    The measured time is end to end around `solve()`: distance computation, worker spawning and
    initialization included, so it compares with the adapters' timed conversions. A multi-worker
    run (`n_workers` > 1) uses the parallel solver with dynamic worker groups and an end-to-end
    budget. `concurrency` runs that many single-worker solves side by side in separate processes;
    it is rejected for multi-worker runs, whose workers already fill the cores.

    Args:
        problem_name: Generator name recorded in each record (e.g. ``"U1"``).
        size: Generator size parameter, recorded in each record.
        time_budgets_sec: Wall-clock budgets in seconds (may be combined with iteration budgets).
        iteration_budgets: Iteration-count budgets (recorded with an ``iterations:`` budget tag);
            single-worker runs only.
        seeds: One independent solve per seed per budget.

    Returns:
        One record per (budget, seed).

    Raises:
        ValueError: If a multi-worker series is combined with iteration budgets or with `concurrency` > 1.
    """
    if n_workers > 1 and (iteration_budgets or concurrency > 1):
        raise ValueError("A multi-worker series takes wall-clock budgets only and runs one solve at a time.")
    budgets: list[tuple[str, TargetDuration]] = []
    for t in time_budgets_sec or []:
        budgets.append((f"time:{t}s", seconds(t)))
    for i in iteration_budgets or []:
        budgets.append((f"iterations:{i}", iterations(i)))
    jobs = [(budget_tag, target, seed) for budget_tag, target in budgets for seed in seeds]

    if concurrency > 1:
        # spawn, not fork: numba's threading layer is not fork-safe
        with ProcessPoolExecutor(max_workers=concurrency, mp_context=get_context("spawn")) as pool:
            results = list(pool.map(_solve_job, [(problem, tag, target, seed, preset, n_workers) for tag, target, seed in jobs]))
    else:
        results = [_solve_job((problem, tag, target, seed, preset, n_workers)) for tag, target, seed in jobs]

    label = maxdiv_tool_label(preset, n_workers)
    return [
        RunRecord(
            tool=label,
            problem=problem_name,
            size=size,
            n=problem.n,
            k=problem.k,
            diversity_metric=problem.diversity_metric.name,
            seed=seed,
            budget=budget_tag,
            measured_sec=measured_sec,
            n_iterations=n_iterations,
            quality=evaluate_selection(problem, i_selected),
            n_constraints=problem.m,
            n_constraints_satisfied=n_constraints_satisfied(problem, i_selected),
        )
        for (budget_tag, _target, seed), (i_selected, measured_sec, n_iterations) in zip(jobs, results)
    ]


def _solve_job(job: tuple) -> tuple:
    """Run one solve and return (selection, end-to-end seconds, iterations); picklable for the process pool."""
    problem, _budget_tag, target, seed, preset, n_workers = job
    if n_workers == 1:
        builder = MaxDivSolverBuilder(problem).with_preset(target, preset).with_seed(seed)
    else:
        builder = ParallelMaxDivSolverBuilder(problem).with_seed(seed).with_workers(target, n_workers).with_end_to_end_budget()
    t0 = time.perf_counter()
    solution = builder.build().solve(verbosity=Verbosity.SILENT)
    measured_sec = time.perf_counter() - t0
    return solution.i_selected, measured_sec, solution.duration.n_iterations
