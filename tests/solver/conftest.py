import numpy as np
import pytest

from max_div.solver import Constraint, DistanceMetric, DiversityMetric, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._strategies import SolverStrategy


@pytest.fixture
def example_problem_1() -> MaxDivSolver:
    # prepare data
    vectors = np.random.rand(10, 5).astype(np.float32)
    selection_size = 5
    init_strategy = SolverStrategy.init_random()
    optim_strategies = [SolverStrategy.optim_dummy(), SolverStrategy.optim_dummy()]
    constraints = [
        Constraint(set(range(0, 5)), min_count=2, max_count=3),
        Constraint(set(range(5, 10)), min_count=2, max_count=3),
    ]

    builder = MaxDivSolverBuilder()

    # return built solver
    builder = (
        builder.with_vectors(vectors)
        .with_selection_size(selection_size)
        .set_initialization_strategy(init_strategy)
        .add_optimization_strategies(optim_strategies)
        .with_distance_metric(DistanceMetric.L1_MANHATTAN)
        .with_diversity_metric(DiversityMetric.min_separation())
        .with_constraints(constraints)
    )
    return builder.build()
