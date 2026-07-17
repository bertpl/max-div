"""Mean/geomean-of-NN reference via the nearest-neighbor assignment model on CP-SAT.

Same Lei & Church-style model as ``scip_nn_separation`` (binary selection x_j, binary
NN-assignment y_ij, closest-assignment constraints), rebuilt on CP-SAT with the objective
weights scaled to integers (CP-SAT is integer-only; the scale bounds the relative
quantization error). CP-SAT with parallel workers is the strongest backend found for this
model, but the formulation's weak relaxation still caps proven optimality around n = 90 —
above that, results are incumbents (best solution found), not certified optima.
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

WEIGHT_SCALE = 1_000_000  # integer weight quantization: ~1e-6 relative resolution


@dataclass
class CpsatNnAssignmentResult:
    """Outcome of one CP-SAT assignment-model solve."""

    i_selected: NDArray[np.int64]
    objective_value: float  # mean of NN separations (geomean for the geomean objective)
    objective_bound: float  # proven upper bound in the same units (== value when optimal)
    proven_optimal: bool
    measured_sec: float


def solve_nn_assignment_cpsat(
    problem: MaxDivProblem,
    metric: DiversityMetric,
    time_limit_sec: float = 120.0,
    num_workers: int = 8,
) -> CpsatNnAssignmentResult:
    """Solve mean- or geomean-of-NN selection (with group constraints) via CP-SAT.

    Args:
        problem: Problem to solve; its constraints are encoded directly.
        metric: MEAN_SEPARATION or GEOMEAN_SEPARATION.
        time_limit_sec: CP-SAT wall-clock limit.
        num_workers: CP-SAT parallel workers (>1 is markedly stronger on this model, at the
            cost of run-to-run determinism).

    Returns:
        Best selection found, its objective value and proven bound, and whether optimality
        was certified.

    Raises:
        ValueError: For unsupported metrics.
        RuntimeError: If CP-SAT finds no feasible selection within the limit.
    """
    from ortools.sat.python import cp_model

    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric for the NN-assignment model: {metric}.")

    t_start = time.perf_counter()
    distances = squareform(problem.condensed_distances().astype(np.float64))
    n, k = problem.n, problem.k
    weights = np.log(np.maximum(distances, 1e-12)) if metric == DiversityMetric.GEOMEAN_SEPARATION else distances
    int_weights = np.round(weights * WEIGHT_SCALE).astype(np.int64)

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{j}") for j in range(n)]
    y = {(i, j): model.new_bool_var(f"y{i}_{j}") for i in range(n) for j in range(n) if i != j}

    model.add(sum(x) == k)
    for i in range(n):
        model.add(sum(y[i, j] for j in range(n) if j != i) == x[i])
        for j in range(n):
            if i != j:
                model.add_implication(y[i, j], x[j])
    _add_closest_assignment_constraints(model, x, y, distances)
    for con in problem.constraints:
        group_sum = sum(x[i] for i in con.int_set)
        model.add(group_sum >= con.min_count)
        model.add(group_sum <= con.max_count)
    model.maximize(sum(int(int_weights[i, j]) * y[i, j] for (i, j) in y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_workers = num_workers
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT found no feasible selection within the time limit.")

    selection = np.asarray([j for j in range(n) if solver.value(x[j])], dtype=np.int64)
    value = _objective_from_log_scale(solver.objective_value, k, metric)
    bound = _objective_from_log_scale(solver.best_objective_bound, k, metric)
    return CpsatNnAssignmentResult(
        i_selected=np.sort(selection),
        objective_value=value,
        objective_bound=bound,
        proven_optimal=status == cp_model.OPTIMAL,
        measured_sec=time.perf_counter() - t_start,
    )


def _objective_from_log_scale(scaled_sum: float, k: int, metric: DiversityMetric) -> float:
    """Convert a scaled objective sum back to a mean (geomean via exp of the log-mean)."""
    mean = scaled_sum / WEIGHT_SCALE / k
    return float(math.exp(mean)) if metric == DiversityMetric.GEOMEAN_SEPARATION else float(mean)


def _add_closest_assignment_constraints(model, x, y, distances) -> None:  # noqa: ANN001 -- cp_model types are dynamic
    """Force each assignment to the *nearest* selected neighbor (same scheme as the SCIP model)."""
    order = np.argsort(distances, axis=1)
    for i in range(distances.shape[0]):
        closer: list = []
        for j_raw in order[i]:
            j = int(j_raw)
            if j == i:
                continue
            model.add(sum(y[i, ll] for ll in closer) + y[i, j] >= x[j] + x[i] - 1)
            closer.append(j)
