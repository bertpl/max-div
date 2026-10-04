import numpy as np

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.solver._solver_state import SolverState
from tests._core.solver.objectives import full_matrix_reader, simple_objective


def new_solver_state(has_constraints: bool) -> SolverState:
    constraints = []
    if has_constraints:
        # Constraint 1: exactly 10 items from indices 0...49
        constraints.append(Constraint(int_set=set(range(50)), min_count=10, max_count=10))
        # Constraint 2: exactly 40 items from indices 50...99
        constraints.append(Constraint(int_set=set(range(50, 100)), min_count=40, max_count=40))

    # Create random 100x5 array
    np.random.seed(42)  # For reproducibility
    vectors = np.random.rand(100, 5).astype(np.float32)

    return SolverState.new(
        n=vectors.shape[0],
        data_matrix_reader=full_matrix_reader(vectors, DistanceMetric.l2_euclidean()),
        k=50,
        diversity_objectives=[simple_objective(DiversityMetric.GEOMEAN_SEPARATION)],
        constraints=constraints,
    )


def new_solver_state_unconstrained(
    n: int = 300, k: int = 30, metric: DiversityMetric = DiversityMetric.MIN_SEPARATION
) -> SolverState:
    """Build a small unconstrained state over precomputed distances, with `metric` as its objective."""
    vectors = np.random.default_rng(20260901).random((n, 3)).astype(np.float32)
    return SolverState.new(
        n=n,
        data_matrix_reader=full_matrix_reader(vectors, DistanceMetric.l2_euclidean()),
        k=k,
        diversity_objectives=[simple_objective(metric)],
        constraints=[],
    )
