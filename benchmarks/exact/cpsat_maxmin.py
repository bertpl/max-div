"""Exact max-min (p-dispersion) reference via CP-SAT threshold binary search.

The max-min optimum equals the largest threshold D for which an "independent set" of
size k exists in the conflict graph {(i, j): d_ij < D}. Each feasibility check is a
boolean CP-SAT model (pair clauses + cardinality + group-count constraints); binary
search over the sorted distinct pairwise distances locates the optimum. Optimality is
proven only when every visited feasibility check returns a conclusive SAT/UNSAT within
the time budget.
"""

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import squareform

from max_div.problem import MaxDivProblem


@dataclass
class CpsatMaxMinResult:
    """Outcome of the threshold binary search.

    ``i_selected`` is the best selection found (from the highest feasible threshold);
    ``proven_optimal`` is True only if the search bracketed the optimum conclusively.
    """

    i_selected: NDArray[np.int64]
    min_separation: float
    proven_optimal: bool
    measured_sec: float
    n_feasibility_solves: int


def solve_maxmin_cpsat(problem: MaxDivProblem, time_limit_sec: float = 60.0) -> CpsatMaxMinResult:
    """Solve max-min selection (with group constraints) to proven optimality via CP-SAT.

    Args:
        problem: Problem to solve; its constraints are encoded directly.
        time_limit_sec: Overall wall-clock budget across all feasibility solves.

    Returns:
        The best selection found and whether optimality was certified.

    Raises:
        RuntimeError: If not even the lowest threshold yields a feasible selection
            within the time budget (no selection to report).
    """
    t_start = time.perf_counter()
    distances = squareform(problem.condensed_distances().astype(np.float64))
    thresholds = np.unique(problem.condensed_distances())  # ascending candidate optima

    # Invariant: threshold index `lo` is known-feasible (its selection is `best`),
    # everything above `hi` is known-infeasible. Optimum = highest feasible threshold.
    lo, hi = 0, len(thresholds) - 1
    best: NDArray[np.int64] | None = None
    conclusive = True

    # The lowest threshold admits any k items satisfying the group constraints; if even
    # that is infeasible (or unsolved in time), there is nothing to report.
    budget = _remaining(t_start, time_limit_sec)
    feasible, selection = _check_threshold(problem, distances, float(thresholds[lo]), budget)
    n_solves = 1
    if not feasible or selection is None:
        raise RuntimeError("CP-SAT could not find any constraint-satisfying selection at the lowest threshold.")
    best = selection

    while lo < hi:
        mid = (lo + hi + 1) // 2
        remaining = _remaining(t_start, time_limit_sec)
        if remaining <= 0:
            conclusive = False
            break
        feasible, selection = _check_threshold(problem, distances, float(thresholds[mid]), remaining)
        n_solves += 1
        if feasible is None:  # solver hit its per-step limit without an answer
            conclusive = False
            break
        if feasible:
            lo, best = mid, selection
        else:
            hi = mid - 1

    return CpsatMaxMinResult(
        i_selected=np.sort(best),
        min_separation=float(thresholds[lo]),
        proven_optimal=conclusive,
        measured_sec=time.perf_counter() - t_start,
        n_feasibility_solves=n_solves,
    )


def _check_threshold(
    problem: MaxDivProblem,
    distances: NDArray[np.float64],
    threshold: float,
    time_limit_sec: float,
) -> tuple[bool | None, NDArray[np.int64] | None]:
    """Ask CP-SAT whether k items with pairwise distance >= threshold satisfy all constraints.

    Returns:
        ``(True, selection)`` if feasible, ``(False, None)`` if proven infeasible, and
        ``(None, None)`` if the solver hit the time limit without a conclusive answer.
    """
    from ortools.sat.python import cp_model

    n, k = problem.n, problem.k
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(n)]

    conflict_i, conflict_j = np.nonzero(np.triu(distances < threshold, k=1))
    for i, j in zip(conflict_i.tolist(), conflict_j.tolist()):
        model.add_bool_or([x[i].negated(), x[j].negated()])

    model.add(sum(x) == k)
    for con in problem.constraints:
        group_sum = sum(x[i] for i in con.int_set)
        model.add(group_sum >= con.min_count)
        model.add(group_sum <= con.max_count)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit_sec, 0.01)
    solver.parameters.num_workers = 1  # deterministic, and comparable across machines
    status = solver.solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        selection = np.asarray([i for i in range(n) if solver.value(x[i])], dtype=np.int64)
        return True, selection
    if status == cp_model.INFEASIBLE:
        return False, None
    return None, None


def _remaining(t_start: float, time_limit_sec: float) -> float:
    """Wall-clock budget left from an overall limit."""
    return time_limit_sec - (time.perf_counter() - t_start)
