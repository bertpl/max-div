"""Run max-div across a budget series, one independent solve per budget x seed, with one or several workers."""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from numpy.typing import NDArray

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


def budget_tag(budget_sec: float) -> str:
    """Return the record tag of a wall-clock budget."""
    return f"time:{budget_sec}s"


def budget_sec(tag: str) -> float | None:
    """Return the wall-clock budget a record tag names, or None for an iteration or single-shot tag."""
    return float(tag.removeprefix("time:").removesuffix("s")) if tag.startswith("time:") else None


@dataclass(frozen=True)
class _SolveJob:
    """One solve to run: the problem, its budget, its seed, and the solver setup."""

    problem: MaxDivProblem
    budget_tag: str
    target: TargetDuration
    seed: int
    preset: SolverPreset
    n_workers: int


@dataclass(frozen=True)
class _SolveOutcome:
    """What one solve produced: the selection, its end-to-end wall-clock, and its iteration count."""

    i_selected: NDArray[np.integer]
    measured_sec: float
    n_iterations: int | None


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

    The measured time is end to end around `solve()`, distance computation, worker spawning and
    initialization included, so it compares with the adapters' timed conversions.

    Args:
        problem_name: Generator name recorded in each record (e.g. ``"U1"``).
        size: Generator size parameter, recorded in each record.
        time_budgets_sec: Wall-clock budgets in seconds (may be combined with iteration budgets).
        iteration_budgets: Iteration-count budgets (recorded with an ``iterations:`` budget tag);
            single-worker runs only.
        seeds: One independent solve per seed per budget.
        n_workers: Above 1, the parallel solver runs this many workers under an end-to-end budget.
        concurrency: How many single-worker solves run side by side, in separate processes.

    Raises:
        ValueError: If a multi-worker series is combined with iteration budgets or with `concurrency` > 1,
            since its workers already fill the cores.
    """
    if n_workers > 1 and (iteration_budgets or concurrency > 1):
        raise ValueError("A multi-worker series takes wall-clock budgets only and runs one solve at a time.")
    budgets: list[tuple[str, TargetDuration]] = []
    for t in time_budgets_sec or []:
        budgets.append((budget_tag(t), seconds(t)))
    for i in iteration_budgets or []:
        budgets.append((f"iterations:{i}", iterations(i)))
    jobs = [_SolveJob(problem, tag, target, seed, preset, n_workers) for tag, target in budgets for seed in seeds]

    if concurrency > 1:
        # spawn, not fork: numba's threading layer is not fork-safe
        with ProcessPoolExecutor(max_workers=concurrency, mp_context=get_context("spawn")) as pool:
            outcomes = list(pool.map(_solve, jobs))
    else:
        outcomes = [_solve(job) for job in jobs]

    label = maxdiv_tool_label(preset, n_workers)
    return [
        RunRecord(
            tool=label,
            problem=problem_name,
            size=size,
            n=problem.n,
            k=problem.k,
            diversity_metric=problem.diversity_metric.name,
            seed=job.seed,
            budget=job.budget_tag,
            measured_sec=outcome.measured_sec,
            n_iterations=outcome.n_iterations,
            quality=evaluate_selection(problem, outcome.i_selected),
            n_constraints=problem.m,
            n_constraints_satisfied=n_constraints_satisfied(problem, outcome.i_selected),
        )
        for job, outcome in zip(jobs, outcomes)
    ]


def _solve(job: _SolveJob) -> _SolveOutcome:
    """Run one solve, timed end to end; picklable for the process pool."""
    if job.n_workers == 1:
        builder = MaxDivSolverBuilder(job.problem).with_preset(job.target, job.preset).with_seed(job.seed)
    else:
        builder = (
            ParallelMaxDivSolverBuilder(job.problem)
            .with_seed(job.seed)
            .with_workers(job.target, job.n_workers)
            .with_end_to_end_budget()
        )
    t0 = time.perf_counter()
    solution = builder.build().solve(verbosity=Verbosity.SILENT)
    return _SolveOutcome(solution.i_selected, time.perf_counter() - t0, solution.duration.n_iterations)
