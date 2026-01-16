import numpy as np

from max_div.random import Constraint
from max_div.solver._distance import DistanceMetric
from max_div.solver._diversity import DiversityMetric
from max_div.solver._solver_state import SolverState


def new_solver_state(has_constraints: bool) -> SolverState:
    constraints = []
    if has_constraints:
        # Constraint 1: exactly 10 items from indices 0...49
        constraints.append(Constraint(int_set=set(range(0, 50)), min_count=10, max_count=10))
        # Constraint 2: exactly 40 items from indices 50...99
        constraints.append(Constraint(int_set=set(range(50, 100)), min_count=40, max_count=40))

    # Create random 100x5 array
    np.random.seed(42)  # For reproducibility
    vectors = np.random.rand(100, 5).astype(np.float32)

    return SolverState.new(
        vectors=vectors,
        k=50,
        distance_metric=DistanceMetric.L2_EUCLIDEAN,
        diversity_metric=DiversityMetric.geomean_separation(),
        diversity_tie_breakers=[],
        constraints=constraints,
    )
