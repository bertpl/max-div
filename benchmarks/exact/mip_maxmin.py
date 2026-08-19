"""Max-min selection as a MIP/CP model, for the exact solvers in the ceilings campaign.

One binary variable per item and, for CP-SAT via its existing threshold search, or for the
MIP solvers a big-M constraint per candidate pair — which is why an exact solver's memory
and time ceilings sit where they sit: the model itself is quadratic in n.

All three entry points share one contract: return a valid size-k selection within the
wall-clock budget. With ``first_feasible`` the solver stops at its first (improving)
solution — the fastest standard setting that still produces a valid selection, which is
what the time ceiling measures. Without it the solver optimizes until the budget runs out.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from benchmarks.adapters._inputs import problem_vectors
from max_div.problem import MaxDivProblem


def _pairwise(problem: MaxDivProblem) -> NDArray[np.float64]:
    """The full pairwise distance matrix the models are built from."""
    return squareform(pdist(problem_vectors(problem).astype(np.float64)))


def solve_maxmin_cpsat_selection(
    problem: MaxDivProblem, budget_sec: float, first_feasible: bool, seed: int
) -> NDArray[np.int64]:
    """CP-SAT: pick k items maximizing the minimum pairwise distance.

    Modeled directly as a satisfaction/optimization problem over scaled integer distances:
    a pair closer than the objective threshold cannot be jointly selected. With
    ``first_feasible`` the search stops at the first solution found.
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
    solver.parameters.max_time_in_seconds = max(budget_sec, 0.01)
    solver.parameters.random_seed = seed
    if first_feasible:
        solver.parameters.stop_after_first_solution = True
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT returned no solution (status {status})")
    return np.asarray([i for i in range(n) if solver.value(x[i])], dtype=np.int64)


def solve_maxmin_scip(
    problem: MaxDivProblem, budget_sec: float, first_feasible: bool, seed: int
) -> NDArray[np.int64]:
    """SCIP on the big-M max-min MIP; ``first_feasible`` stops at the first solution."""
    from pyscipopt import Model, quicksum

    dist = _pairwise(problem)
    n, k = problem.n, problem.k
    big_m = float(dist.max())

    model = Model()
    model.hideOutput()
    model.setParam("limits/time", max(budget_sec, 0.01))
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
    model.optimize()
    if model.getNSols() == 0:
        raise RuntimeError("SCIP returned no solution within the budget")
    sol = model.getBestSol()
    return np.asarray([i for i in range(n) if model.getSolVal(sol, x[i]) > 0.5], dtype=np.int64)


def solve_maxmin_highs(
    problem: MaxDivProblem, budget_sec: float, first_feasible: bool, seed: int
) -> NDArray[np.int64]:
    """HiGHS on the big-M max-min MIP; ``first_feasible`` stops at the first improving solution."""
    import highspy

    dist = _pairwise(problem)
    n, k = problem.n, problem.k
    big_m = float(dist.max())

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", max(budget_sec, 0.01))
    h.setOptionValue("random_seed", seed)
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

    h.run()
    solution = h.getSolution()
    values = np.asarray(solution.col_value[:n])
    selected = np.argsort(values)[-k:]
    if not np.all(values[selected] > 0.5):
        raise RuntimeError("HiGHS returned no integral selection within the budget")
    return np.sort(selected).astype(np.int64)
