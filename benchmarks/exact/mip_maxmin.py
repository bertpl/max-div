"""Max-min selection as a MIP/CP model, for the exact solvers in the solver-scaling benchmarks.

The model carries one binary variable per item and a per-pair constraint (big-M for the
MIP solvers, an enforcement literal for CP-SAT), so it is quadratic in n — which is what
bounds how large a problem an exact solver can handle in memory and in time.

All three entry points share one contract: return a valid size-k selection by ``deadline``,
a ``time.monotonic()`` timestamp. Each solver's internal time limit only covers its solving
phase, so it is set to the time remaining at the moment solving starts — the model
construction before it shrinks the solver's budget instead of adding to the measured
end-to-end time on top of it. With ``first_feasible`` the solver stops at its first
(improving) solution — the fastest standard setting that still produces a valid selection;
without it the solver optimizes until its time runs out.
"""

import time

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from benchmarks.adapters._inputs import problem_vectors
from max_div.problem import MaxDivProblem


def _pairwise(problem: MaxDivProblem) -> NDArray[np.float64]:
    """Return the full pairwise distance matrix the models are built from."""
    return squareform(pdist(problem_vectors(problem).astype(np.float64)))


def _remaining_sec(deadline: float) -> float:
    """Return the time left until `deadline` (a `time.monotonic()` value), floored at 0.01 s."""
    return max(deadline - time.monotonic(), 0.01)


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
    from pyscipopt import Model, quicksum

    dist = _pairwise(problem)
    n, k = problem.n, problem.k
    big_m = float(dist.max())

    model = Model()
    model.hideOutput()
    model.setParam("randomization/randomseedshift", seed)
    if first_feasible:
        model.setParam("limits/solutions", 1)
    x = [model.addVar(vtype="B", name=f"x{i}") for i in range(n)]
    t = model.addVar(vtype="C", lb=0.0, ub=big_m, name="t")
    model.addCons(quicksum(x) == k)
    for i in range(n):
        for j in range(i + 1, n):
            model.addCons(t <= dist[i, j] + big_m * (2 - x[i] - x[j]))
    model.setObjective(t, "maximize")
    model.setParam("limits/time", _remaining_sec(deadline))
    model.optimize()
    if model.getNSols() == 0:
        raise RuntimeError("SCIP returned no solution within the budget")
    sol = model.getBestSol()
    return np.asarray([i for i in range(n) if model.getSolVal(sol, x[i]) > 0.5], dtype=np.int64)


def solve_maxmin_highs_selection(
    problem: MaxDivProblem, deadline: float, first_feasible: bool, seed: int, num_workers: int = 1
) -> NDArray[np.int64]:
    """HiGHS on the big-M max-min MIP; ``num_workers`` sets the parallel branch-and-bound threads."""
    import highspy

    dist = _pairwise(problem)
    n, k = problem.n, problem.k
    big_m = float(dist.max())

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("random_seed", seed)
    h.setOptionValue("threads", num_workers)
    if first_feasible:
        h.setOptionValue("mip_max_improving_sols", 1)

    # Variables: x_0..x_{n-1} binary, then t continuous. Objective: maximize t.
    inf = highspy.kHighsInf
    h.addVars(n, np.zeros(n), np.ones(n))
    h.addVar(0.0, big_m)
    for i in range(n):
        h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
    h.changeColCost(n, -1.0)  # HiGHS minimizes; -t makes it maximize t

    # sum(x) == k
    h.addRow(k, k, n, np.arange(n, dtype=np.int32), np.ones(n))
    # t <= d_ij + M(2 - x_i - x_j)  <=>  t + M*x_i + M*x_j <= d_ij + 2M
    for i in range(n):
        for j in range(i + 1, n):
            idx = np.asarray([i, j, n], dtype=np.int32)
            coef = np.asarray([big_m, big_m, 1.0])
            h.addRow(-inf, dist[i, j] + 2 * big_m, 3, idx, coef)

    h.setOptionValue("time_limit", _remaining_sec(deadline))
    h.run()
    values = np.asarray(h.getSolution().col_value[:n])
    # a valid answer needs exactly k variables at 1; a fractional or wrong-count solution (no
    # integer solution found within the budget) is not a selection we can report
    selected = np.flatnonzero(values > 0.5)
    if len(selected) != k:
        raise RuntimeError("HiGHS found no integral k-selection within the budget")
    return np.sort(selected).astype(np.int64)
