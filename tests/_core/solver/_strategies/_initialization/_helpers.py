import numpy as np

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._solver_state import SolverState
from tests._core.solver.objectives import single_term_objective


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
        store=DistanceStore.full_matrix_from_vectors(vectors, DistanceMetric.l2_euclidean()),
        k=50,
        diversity_objective=single_term_objective(DiversityMetric.GEOMEAN_SEPARATION),
        diversity_tie_breakers=[],
        constraints=constraints,
    )


def new_solver_state_unconstrained(n: int = 300, k: int = 30) -> SolverState:
    """Build a small unconstrained state over precomputed distances."""
    vectors = np.random.default_rng(20260901).random((n, 3)).astype(np.float32)
    return SolverState.new(
        n=n,
        store=DistanceStore.full_matrix_from_vectors(vectors, DistanceMetric.l2_euclidean()),
        k=k,
        diversity_objective=single_term_objective(DiversityMetric.MIN_SEPARATION),
        diversity_tie_breakers=[],
        constraints=[],
    )
