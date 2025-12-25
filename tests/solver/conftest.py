import numpy as np
import pytest

from max_div.solver import Constraint, DistanceMetric, DiversityMetric, MaxDivProblem, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._duration import iterations, seconds
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


@pytest.fixture
def example_problem_1() -> MaxDivSolver:
    # prepare data
    vectors = np.random.rand(10, 5).astype(np.float32)
    selection_size = 5
    init_strategy = InitializationStrategy.random_one_shot()
    solver_steps = [
        OptimizationStep(OptimizationStrategy.random_swaps(), seconds(0.1)),
        OptimizationStep(OptimizationStrategy.random_swaps(), iterations(1234)),
    ]
    constraints = [
        Constraint(set(range(0, 5)), min_count=2, max_count=3),
        Constraint(set(range(5, 10)), min_count=2, max_count=3),
    ]

    # return built solver
    builder = (
        MaxDivSolverBuilder(
            MaxDivProblem(
                vectors=vectors,
                k=selection_size,
                distance_metric=DistanceMetric.L1_MANHATTAN,
                diversity_metric=DiversityMetric.min_separation(),
                constraints=constraints,
            )
        )
        .set_initialization_strategy(init_strategy)
        .add_solver_steps(solver_steps)
    )
    return builder.build()
