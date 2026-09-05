from itertools import combinations
from multiprocessing import get_context

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from benchmarks.common import evaluate_selection
from benchmarks.exact import (
    solve_maxmin_cpsat,
    solve_maxmin_highs,
    solve_maxmin_scip,
    solve_nn_assignment_cpsat,
    solve_nn_separation_scip,
)
from max_div._core.constraints import Constraint
from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem


def _tiny_problem(metric: DiversityMetric) -> MaxDivProblem:
    """15 vectors, select 4, one group constraint — small enough to brute-force."""
    rng = np.random.default_rng(11)
    return MaxDivProblem.new(
        vectors=rng.random((15, 2)).astype(np.float32),
        k=4,
        diversity_metric=metric,
        constraints=[Constraint(int_set=set(range(8)), min_count=1, max_count=2)],
    )


def _brute_force_optimum(problem: MaxDivProblem, metric_name: str) -> float:
    """Enumerate all feasible selections and return the true optimum (the test oracle)."""
    best = -np.inf
    dist = squareform(problem.condensed_distances().astype(np.float64))
    group = problem.constraints[0].int_set
    for selection in combinations(range(problem.n), problem.k):
        if not 1 <= len(group.intersection(selection)) <= 2:
            continue
        value = evaluate_selection(problem, np.asarray(selection, dtype=np.int64))[metric_name]
        best = max(best, value)
    assert dist.shape[0] == problem.n  # oracle sanity: distances cover the ground set
    return best


def test_cpsat_maxmin_matches_brute_force():
    # --- arrange -----------------------------------------
    problem = _tiny_problem(DiversityMetric.MIN_SEPARATION)
    oracle = _brute_force_optimum(problem, "MIN_SEPARATION")

    # --- act ---------------------------------------------
    result = solve_maxmin_cpsat(problem, time_limit_sec=30)
    achieved = evaluate_selection(problem, result.i_selected)["MIN_SEPARATION"]

    # --- assert ------------------------------------------
    assert result.proven_optimal
    assert achieved == pytest.approx(oracle, rel=1e-6)


@pytest.mark.parametrize("metric", [DiversityMetric.MEAN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION])
def test_scip_nn_separation_matches_brute_force(metric):
    # --- arrange -----------------------------------------
    problem = _tiny_problem(metric)
    oracle = _brute_force_optimum(problem, metric.name)

    # --- act ---------------------------------------------
    result = solve_nn_separation_scip(problem, metric, time_limit_sec=60)
    achieved = evaluate_selection(problem, result.i_selected)[metric.name]

    # --- assert ------------------------------------------
    assert result.proven_optimal
    assert achieved == pytest.approx(oracle, rel=1e-5)


def test_scip_rejects_unsupported_metric():
    # --- arrange -----------------------------------------
    problem = _tiny_problem(DiversityMetric.MIN_SEPARATION)

    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="Unsupported metric"):
        solve_nn_separation_scip(problem, DiversityMetric.MIN_SEPARATION)


@pytest.mark.parametrize("metric", [DiversityMetric.MEAN_SEPARATION, DiversityMetric.GEOMEAN_SEPARATION])
def test_cpsat_nn_assignment_matches_brute_force(metric):
    # the CP-SAT rebuild of the assignment model must agree with the brute-force oracle,
    # which also validates its integer weight scaling end to end
    # --- arrange -----------------------------------------
    problem = _tiny_problem(metric)
    oracle = _brute_force_optimum(problem, metric.name)

    # --- act ---------------------------------------------
    result = solve_nn_assignment_cpsat(problem, metric, time_limit_sec=60, num_workers=1)
    achieved = evaluate_selection(problem, result.i_selected)[metric.name]

    # --- assert ------------------------------------------
    assert result.proven_optimal
    assert achieved == pytest.approx(oracle, rel=1e-5)
    assert result.objective_bound == pytest.approx(result.objective_value, rel=1e-5)


def test_cpsat_nn_assignment_rejects_unsupported_metric():
    # --- arrange -----------------------------------------
    problem = _tiny_problem(DiversityMetric.MIN_SEPARATION)

    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="Unsupported metric"):
        solve_nn_assignment_cpsat(problem, DiversityMetric.MIN_SEPARATION)


def _mip_problem() -> MaxDivProblem:
    """12 vectors, select 3, unconstrained — small enough to brute-force."""
    rng = np.random.default_rng(12)
    return MaxDivProblem.new(vectors=rng.random((12, 2)).astype(np.float32), k=3)


def _mip_oracle(problem: MaxDivProblem) -> float:
    """Enumerate all selections and return the true max-min optimum."""
    return max(
        evaluate_selection(problem, np.asarray(s, dtype=np.int64))["MIN_SEPARATION"]
        for s in combinations(range(problem.n), problem.k)
    )


def _solve_highs_in_child(connection) -> None:  # noqa: ANN001 -- multiprocessing connection
    """Child-process body: HiGHS cannot load after SCIP in one process (a shared-library clash), so it runs apart."""
    connection.send(solve_maxmin_highs(_mip_problem(), time_limit_sec=60))
    connection.close()


def test_scip_maxmin_certifies_the_brute_force_optimum():
    """SCIP certifies the unconstrained max-min optimum and reports it with the selection."""
    # --- arrange -----------------------------------------
    problem = _mip_problem()
    oracle = _mip_oracle(problem)

    # --- act ---------------------------------------------
    result = solve_maxmin_scip(problem, time_limit_sec=60)

    # --- assert ------------------------------------------
    assert result.proven_optimal
    assert result.min_separation == pytest.approx(oracle, rel=1e-5)
    assert evaluate_selection(problem, result.i_selected)["MIN_SEPARATION"] == pytest.approx(oracle, rel=1e-5)


def test_highs_maxmin_certifies_the_brute_force_optimum():
    """HiGHS certifies the unconstrained max-min optimum, run in its own process as the harness does."""
    # --- arrange -----------------------------------------
    problem = _mip_problem()
    oracle = _mip_oracle(problem)
    context = get_context("spawn")
    parent, child = context.Pipe(duplex=False)

    # --- act ---------------------------------------------
    process = context.Process(target=_solve_highs_in_child, args=(child,))
    process.start()
    child.close()
    result = parent.recv()
    process.join()

    # --- assert ------------------------------------------
    assert result.proven_optimal
    assert result.min_separation == pytest.approx(oracle, rel=1e-5)
    assert evaluate_selection(problem, result.i_selected)["MIN_SEPARATION"] == pytest.approx(oracle, rel=1e-5)
