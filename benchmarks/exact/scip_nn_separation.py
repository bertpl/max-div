"""Exact mean/geomean-of-NN reference via a nearest-neighbor assignment MILP (SCIP).

Model (Lei & Church-style): binary x_j = "item j selected", binary y_ij = "j is i's
nearest selected neighbor". Each selected item is assigned exactly one neighbor, and
closest-assignment constraints force that neighbor to be the *nearest* selected one.
Maximizing sum(d_ij * y_ij) then solves mean-of-NN separation exactly; running the same
model on log-distances solves geomean-of-NN (the geometric mean is the arithmetic mean
after a log transform). Group min/max-count constraints bolt on directly. O(n^2)
binaries: practical only at small n — that is the point of an exact reference.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import squareform

from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem

SUPPORTED_METRICS = (DiversityMetric.MEAN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION)


@dataclass
class ScipNnSeparationResult:
    """Outcome of the assignment-MILP solve."""

    i_selected: NDArray[np.int64]
    objective_value: float  # mean of NN separations (geomean for the geomean objective)
    proven_optimal: bool
    measured_sec: float


def solve_nn_separation_scip(
    problem: MaxDivProblem,
    metric: DiversityMetric,
    time_limit_sec: float = 120.0,
) -> ScipNnSeparationResult:
    """Solve mean- or geomean-of-NN selection (with group constraints) via SCIP.

    Args:
        problem: Problem to solve; its constraints are encoded directly.
        metric: MEAN_SEPARATION or GEOMEAN_SEPARATION.
        time_limit_sec: SCIP wall-clock limit.

    Returns:
        Best selection found, its objective value, and whether optimality was proven.

    Raises:
        ValueError: For unsupported metrics.
        RuntimeError: If SCIP finds no feasible selection within the limit.
    """
    from pyscipopt import Model, quicksum

    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric for the NN-assignment MILP: {metric}.")

    t_start = time.perf_counter()
    distances = squareform(problem.condensed_distances().astype(np.float64))
    n, k = problem.n, problem.k
    weights = np.log(np.maximum(distances, 1e-12)) if metric == DiversityMetric.GEOMEAN_SEPARATION else distances

    model = Model("nn-separation")
    model.hideOutput()
    model.setParam("limits/time", time_limit_sec)

    x = [model.addVar(vtype="B", name=f"x{j}") for j in range(n)]
    y = {(i, j): model.addVar(vtype="B", name=f"y{i}_{j}") for i in range(n) for j in range(n) if i != j}

    model.addCons(quicksum(x) == k)
    _add_assignment_constraints(model, x, y, n)
    _add_closest_assignment_constraints(model, x, y, distances)
    for con in problem.constraints:
        group_sum = quicksum(x[i] for i in con.int_set)
        model.addCons(group_sum >= con.min_count)
        model.addCons(group_sum <= con.max_count)

    model.setObjective(quicksum(weights[i, j] * y[i, j] for (i, j) in y), "maximize")
    model.optimize()

    if model.getNSols() == 0:
        raise RuntimeError("SCIP found no feasible selection within the time limit.")

    selection = np.asarray([j for j in range(n) if model.getVal(x[j]) > 0.5], dtype=np.int64)
    obj = model.getObjVal() / k  # mean of NN contributions (log-domain for geomean)
    if metric == DiversityMetric.GEOMEAN_SEPARATION:
        obj = math.exp(obj)

    return ScipNnSeparationResult(
        i_selected=np.sort(selection),
        objective_value=float(obj),
        proven_optimal=model.getStatus() == "optimal",
        measured_sec=time.perf_counter() - t_start,
    )


def _add_assignment_constraints(model, x, y, n) -> None:  # noqa: ANN001 -- pyscipopt types are dynamic
    """Each selected item gets exactly one NN assignment, and only to a selected neighbor."""
    from pyscipopt import quicksum

    for i in range(n):
        model.addCons(quicksum(y[i, j] for j in range(n) if j != i) == x[i])
        for j in range(n):
            if i != j:
                model.addCons(y[i, j] <= x[j])


def _add_closest_assignment_constraints(model, x, y, distances) -> None:  # noqa: ANN001 -- pyscipopt types are dynamic
    """Force each assignment to the *nearest* selected neighbor.

    If i and j are both selected, i must be assigned to some neighbor at least as
    close as j; walking neighbors in distance order accumulates the "closer" set.
    """
    from pyscipopt import quicksum

    order = np.argsort(distances, axis=1)
    for i in range(distances.shape[0]):
        closer: list = []
        for j_raw in order[i]:
            j = int(j_raw)
            if j == i:
                continue
            model.addCons(quicksum(y[i, ll] for ll in closer) + y[i, j] >= x[j] + x[i] - 1)
            closer.append(j)
