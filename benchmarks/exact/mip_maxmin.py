"""Max-min selection as a MIP/CP model, for the exact solvers of the scaling and head-to-head benchmarks.

The model carries one binary variable per item, a per-pair constraint (big-M for the
MIP solvers, an enforcement literal for CP-SAT) and one count row per fairness constraint, so
it is quadratic in n — which is what bounds how large a problem an exact solver can handle in
memory and in time.

Two families of entry points share the model builders. The ``*_selection`` functions serve the
scaling benchmarks: return a valid size-k selection by ``deadline``, a ``time.monotonic()``
timestamp, where each solver's internal time limit only covers its solving phase and is set to
the time remaining at the moment solving starts — model construction shrinks the solver's budget
instead of adding to the measured end-to-end time. With ``first_feasible`` the solver stops at its
first (improving) solution; without it the solver optimizes until its time runs out. The
certifying functions serve the head-to-head comparison: run to proven optimality or the time
limit and report the optimum with its proof status.
"""

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from benchmarks.adapters._inputs import problem_vectors
from max_div.problem import MaxDivProblem


def _pairwise(problem: MaxDivProblem) -> NDArray[np.float64]:
    """Return the full pairwise distance matrix the models are built from."""
    return squareform(pdist(problem_vectors(problem).astype(np.float64)))


def _remaining_sec(deadline: float) -> float:
    """Return the time left until `deadline` (a `time.monotonic()` value), clamped to a small positive value."""
    return max(deadline - time.monotonic(), 0.01)


# ==================================================================================================
#  Model builders
# ==================================================================================================
def _scip_model(problem: MaxDivProblem, seed: int) -> tuple:
    """Build the big-M max-min MIP on SCIP; return (model, x, t) with output hidden and the seed set."""
    from pyscipopt import Model, quicksum

    dist = _pairwise(problem)
    n, k = problem.n, problem.k
    big_m = float(dist.max())

    model = Model()
    model.hideOutput()
    model.setParam("randomization/randomseedshift", seed)
    x = [model.addVar(vtype="B", name=f"x{i}") for i in range(n)]
    t = model.addVar(vtype="C", lb=0.0, ub=big_m, name="t")
    model.addCons(quicksum(x) == k)
    for con in problem.constraints:
        group_sum = quicksum(x[i] for i in sorted(con.int_set))
        model.addCons(group_sum >= con.min_count)
        model.addCons(group_sum <= con.max_count)
    for i in range(n):
        for j in range(i + 1, n):
            model.addCons(t <= dist[i, j] + big_m * (2 - x[i] - x[j]))
    model.setObjective(t, "maximize")
    return model, x, t


def _scip_selection(model, x, n: int) -> NDArray[np.int64]:  # noqa: ANN001 -- pyscipopt types are dynamic
    """Return the best SCIP solution's selection.

    Raises:
        RuntimeError: If SCIP holds no solution.
    """
    if model.getNSols() == 0:
        raise RuntimeError("SCIP returned no solution within the time limit")
    sol = model.getBestSol()
    return np.asarray([i for i in range(n) if model.getSolVal(sol, x[i]) > 0.5], dtype=np.int64)


def _highs_model(problem: MaxDivProblem, seed: int, num_workers: int):  # noqa: ANN202 -- highspy types are dynamic
    """Build the big-M max-min MIP on HiGHS; return the solver with output hidden, the seed and threads set."""
    import highspy

    dist = _pairwise(problem)
    n, k = problem.n, problem.k
    big_m = float(dist.max())

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("random_seed", seed)
    h.setOptionValue("threads", num_workers)
    # Variables: x_0..x_{n-1} binary, then t continuous. Objective: maximize t.
    inf = highspy.kHighsInf
    h.addVars(n, np.zeros(n), np.ones(n))
    h.addVar(0.0, big_m)
    for i in range(n):
        h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
    h.changeColCost(n, -1.0)  # HiGHS minimizes; -t makes it maximize t
    # sum(x) == k
    h.addRow(k, k, n, np.arange(n, dtype=np.int32), np.ones(n))
    # min_count <= sum(x_i for i in group) <= max_count, per fairness constraint
    for con in problem.constraints:
        members = np.asarray(sorted(con.int_set), dtype=np.int32)
        h.addRow(con.min_count, con.max_count, len(members), members, np.ones(len(members)))
    # t <= d_ij + M(2 - x_i - x_j)  <=>  t + M*x_i + M*x_j <= d_ij + 2M
    for i in range(n):
        for j in range(i + 1, n):
            idx = np.asarray([i, j, n], dtype=np.int32)
            coef = np.asarray([big_m, big_m, 1.0])
            h.addRow(-inf, dist[i, j] + 2 * big_m, 3, idx, coef)
    return h


def _highs_selection(h, n: int, k: int) -> NDArray[np.int64]:  # noqa: ANN001 -- highspy types are dynamic
    """Return HiGHS's selection: exactly k variables at 1.

    Raises:
        RuntimeError: If no integral k-selection is held — a fractional or wrong-count solution
            is not a selection to report.
    """
    values = np.asarray(h.getSolution().col_value[:n])
    selected = np.flatnonzero(values > 0.5)
    if len(selected) != k:
        raise RuntimeError("HiGHS found no integral k-selection within the time limit")
    return np.sort(selected).astype(np.int64)


# ==================================================================================================
#  Scaling benchmarks: a selection by the deadline
# ==================================================================================================
def solve_maxmin_cpsat_selection(
    problem: MaxDivProblem, deadline: float, first_feasible: bool, seed: int, num_workers: int = 1
) -> NDArray[np.int64]:
    """CP-SAT: pick k items maximizing the minimum pairwise distance.

    Modeled directly as a satisfaction/optimization problem over scaled integer distances: a
    pair closer than the objective threshold cannot be jointly selected. `num_workers` sets the
    portfolio-search parallelism.
    """
    from ortools.sat.python import cp_model

    dist = _pairwise(problem)
    n, k = problem.n, problem.k
    scaled = np.round(dist * 1_000_000).astype(np.int64)

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(x) == k)
    if not first_feasible:
        t = model.new_int_var(0, int(scaled.max()), "t")
        for i in range(n):
            for j in range(i + 1, n):
                model.add(t <= scaled[i, j]).only_enforce_if([x[i], x[j]])
        model.maximize(t)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = _remaining_sec(deadline)
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = num_workers
    if first_feasible:
        solver.parameters.stop_after_first_solution = True
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT returned no solution (status {status})")
    return np.asarray([i for i in range(n) if solver.value(x[i])], dtype=np.int64)


def solve_maxmin_scip_selection(
    problem: MaxDivProblem, deadline: float, first_feasible: bool, seed: int
) -> NDArray[np.int64]:
    """SCIP on the big-M max-min MIP."""
    model, x, _t = _scip_model(problem, seed)
    if first_feasible:
        model.setParam("limits/solutions", 1)
    model.setParam("limits/time", _remaining_sec(deadline))
    model.optimize()
    return _scip_selection(model, x, problem.n)


def solve_maxmin_highs_selection(
    problem: MaxDivProblem, deadline: float, first_feasible: bool, seed: int, num_workers: int = 1
) -> NDArray[np.int64]:
    """HiGHS on the big-M max-min MIP; ``num_workers`` sets the parallel branch-and-bound threads."""
    h = _highs_model(problem, seed, num_workers)
    if first_feasible:
        h.setOptionValue("mip_max_improving_sols", 1)
    h.setOptionValue("time_limit", _remaining_sec(deadline))
    h.run()
    return _highs_selection(h, problem.n, problem.k)


# ==================================================================================================
#  Head-to-head comparison: certified optima
# ==================================================================================================
@dataclass
class MaxMinResult:
    """Record the outcome of one certifying max-min solve: the selection, its objective, and whether it is proven optimal."""

    i_selected: NDArray[np.int64]
    min_separation: float
    proven_optimal: bool
    measured_sec: float


def solve_maxmin_scip(problem: MaxDivProblem, time_limit_sec: float, seed: int = 0) -> MaxMinResult:
    """Solve the big-M max-min MIP with SCIP, to proven optimality or the time limit.

    Raises:
        RuntimeError: If SCIP finds no solution within the limit.
    """
    t_start = time.perf_counter()
    model, x, _t = _scip_model(problem, seed)
    model.setParam("limits/time", max(time_limit_sec - (time.perf_counter() - t_start), 0.01))
    model.optimize()
    selection = _scip_selection(model, x, problem.n)
    return MaxMinResult(
        i_selected=np.sort(selection),
        min_separation=float(model.getSolObjVal(model.getBestSol())),
        proven_optimal=model.getStatus() == "optimal",
        measured_sec=time.perf_counter() - t_start,
    )


def solve_maxmin_highs(problem: MaxDivProblem, time_limit_sec: float, seed: int = 0, num_workers: int = 1) -> MaxMinResult:
    """Solve the big-M max-min MIP with HiGHS, to proven optimality or the time limit.

    Raises:
        RuntimeError: If HiGHS finds no integral k-selection within the limit.
    """
    import highspy

    t_start = time.perf_counter()
    h = _highs_model(problem, seed, num_workers)
    h.setOptionValue("time_limit", max(time_limit_sec - (time.perf_counter() - t_start), 0.01))
    h.run()
    selection = _highs_selection(h, problem.n, problem.k)
    return MaxMinResult(
        i_selected=selection,
        min_separation=float(h.getSolution().col_value[problem.n]),
        proven_optimal=h.getModelStatus() == highspy.HighsModelStatus.kOptimal,
        measured_sec=time.perf_counter() - t_start,
    )
