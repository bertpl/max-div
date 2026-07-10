import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolver, MaxDivSolverBuilder
from max_div._core.solver._duration import iterations, seconds
from max_div._core.solver._solver_step import OptimizationStep
from max_div._core.solver._strategies import InitializationStrategy, OptimizationStrategy


@pytest.fixture
def example_solver() -> MaxDivSolver:
    # prepare data
    vectors = np.random.rand(10, 5).astype(np.float32)
    selection_size = 5
    init_strategy = InitializationStrategy.random_one_shot()
    solver_steps = [
        OptimizationStep(OptimizationStrategy.random_swaps(), seconds(0.1)),
        OptimizationStep(OptimizationStrategy.random_swaps(), iterations(1234)),
    ]
    constraints = [
        Constraint(set(range(5)), min_count=2, max_count=3),
        Constraint(set(range(5, 10)), min_count=2, max_count=3),
    ]

    # return built solver
    builder = (
        MaxDivSolverBuilder(
            MaxDivProblem(
                vectors=vectors,
                k=selection_size,
                distance_metric=DistanceMetric.L1_MANHATTAN,
                diversity_metric=DiversityMetric.MIN_SEPARATION,
                constraints=constraints,
            )
        )
        .set_initialization_strategy(init_strategy)
        .add_solver_steps(solver_steps)
    )
    return builder.build()
