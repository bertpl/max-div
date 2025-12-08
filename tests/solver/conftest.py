import numpy as np
import pytest

from max_div.solver import Constraint, DistanceMetric, DiversityMetric, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._duration import iterations, seconds
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


@pytest.fixture
def example_problem_1() -> MaxDivSolver:
    # prepare data
    vectors = np.random.rand(10, 5).astype(np.float32)
    selection_size = 5
    init_strategy = InitializationStrategy.random()
    solver_steps = [
        OptimizationStep(OptimizationStrategy.dummy(), seconds(0.1)),
        OptimizationStep(OptimizationStrategy.dummy(), iterations(1234)),
    ]
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
        .add_solver_steps(solver_steps)
        .with_distance_metric(DistanceMetric.L1_MANHATTAN)
        .with_diversity_metric(DiversityMetric.min_separation())
        .with_constraints(constraints)
    )
    return builder.build()
