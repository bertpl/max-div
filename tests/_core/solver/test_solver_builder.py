import numpy as np
import pytest

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import (
    ConstraintPenalty,
    MaxDivSolver,
    MaxDivSolverBuilder,
    SolverPreset,
    iterations,
    seconds,
)
from max_div._core.solver._solver_step import InitializationStep, OptimizationStep
from max_div._core.solver._strategies import InitializationStrategy, OptimizationStrategy
from tests.helpers import swept_benchmark_problems


# =================================================================================================
#  Fixture
# =================================================================================================
@pytest.fixture
def dummy_problem() -> MaxDivProblem:
    return MaxDivProblem.new(
        vectors=np.random.rand(10, 5).astype(np.float32),
        k=3,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
    )


# =================================================================================================
#  MaxDivSolverBuilder - Modifiers
# =================================================================================================
@pytest.mark.parametrize(
    "strategy, expected_ok",
    [
        (InitializationStrategy.random_one_shot(), True),
        (OptimizationStrategy.random_swaps(), False),
    ],
)
def test_solver_builder_set_initialization_strategy(dummy_problem, strategy, expected_ok: bool):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder(dummy_problem)

    # --- act & assert ------------------------------------
    if expected_ok:
        builder = builder.set_initialization_strategy(strategy)
        assert builder is not None
    else:
        with pytest.raises(TypeError):
            _ = builder.set_initialization_strategy(strategy)


@pytest.mark.parametrize(
    "strategies, expected_ok",
    [
        ([OptimizationStrategy.random_swaps(), OptimizationStrategy.random_swaps()], True),
        ([OptimizationStrategy.random_swaps()], True),
        ([InitializationStrategy.random_one_shot()], False),
        ([InitializationStrategy.random_one_shot(), OptimizationStrategy.random_swaps()], False),
        ([OptimizationStrategy.random_swaps(), InitializationStrategy.random_one_shot()], False),
    ],
)
def test_solver_builder_add_solver_steps(dummy_problem, strategies: list, expected_ok: bool):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder(dummy_problem)
    solver_steps = [
        InitializationStep(s) if isinstance(s, InitializationStrategy) else OptimizationStep(s, duration=seconds(10))
        for s in strategies
    ]

    # --- act & assert ------------------------------------
    if expected_ok:
        builder = builder.add_solver_steps(solver_steps)
        assert builder is not None
    else:
        with pytest.raises(TypeError):
            _ = builder.add_solver_steps(solver_steps)


# =================================================================================================
#  MaxDivSolverBuilder - Tie-Breaker Metrics
# =================================================================================================
@pytest.mark.parametrize(
    "diversity_metric, expected_tie_breakers",
    [
        (
            DiversityMetric.MIN_SEPARATION,
            [DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC],
        ),
        (DiversityMetric.GEOMEAN_SEPARATION, [DiversityMetric.NON_ZERO_SEPARATION_FRAC]),
        (DiversityMetric.APPROX_GEOMEAN_SEPARATION, [DiversityMetric.NON_ZERO_SEPARATION_FRAC]),
        (DiversityMetric.MEAN_SEPARATION, []),
    ],
)
def test_max_div_solver_builder_tie_breaker_metrics_defaults(
    diversity_metric: DiversityMetric, expected_tie_breakers: list[DiversityMetric]
):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder(
        MaxDivProblem.new(
            vectors=np.random.rand(10, 5).astype(np.float32),
            k=3,
            diversity_metric=diversity_metric,
        )
    ).with_default_diversity_tie_breakers()

    # --- act ---------------------------------------------
    solver = builder.build()

    # --- assert ------------------------------------------
    assert solver._diversity_metric == diversity_metric
    for true_tie_breaker, expected_tie_breaker in zip(solver._diversity_tie_breakers, expected_tie_breakers):
        assert true_tie_breaker == expected_tie_breaker


def test_max_div_solver_builder_tie_breaker_metrics_custom(dummy_problem):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder(dummy_problem).with_diversity_tie_breakers(
        [
            DiversityMetric.APPROX_GEOMEAN_SEPARATION,
            DiversityMetric.NON_ZERO_SEPARATION_FRAC,
            DiversityMetric.MEAN_SEPARATION,
        ]
    )

    # --- act ---------------------------------------------
    solver = builder.build()

    # --- assert ------------------------------------------
    assert solver._diversity_metric == DiversityMetric.GEOMEAN_SEPARATION
    assert len(solver._diversity_tie_breakers) == 3
    assert solver._diversity_tie_breakers[0] == DiversityMetric.APPROX_GEOMEAN_SEPARATION
    assert solver._diversity_tie_breakers[1] == DiversityMetric.NON_ZERO_SEPARATION_FRAC
    assert solver._diversity_tie_breakers[2] == DiversityMetric.MEAN_SEPARATION


# =================================================================================================
#  MaxDivSolverBuilder - End-to-End
# =================================================================================================
def test_max_div_solver_builder_end_to_end():
    # --- arrange -----------------------------------------
    vectors = np.random.rand(10, 5).astype(np.float32)
    k = 5
    init_strategy = InitializationStrategy.random_one_shot()
    solver_steps = [
        OptimizationStep(OptimizationStrategy.random_swaps(), seconds(1)),
        OptimizationStep(OptimizationStrategy.random_swaps(), iterations(100)),
    ]
    constraints = [
        Constraint(set(range(5)), min_count=2, max_count=3),
        Constraint(set(range(5, 10)), min_count=2, max_count=3),
    ]

    # --- act ---------------------------------------------
    builder = (
        MaxDivSolverBuilder(
            MaxDivProblem.new(
                vectors=vectors,
                k=k,
                distance_metric=DistanceMetric.L1_MANHATTAN,
                diversity_metric=DiversityMetric.MIN_SEPARATION,
                constraints=constraints,
            )
        )
        .set_initialization_strategy(init_strategy)
        .add_solver_steps(solver_steps)
        .with_seed(123)
    )
    solver = builder.build()

    # --- assert ------------------------------------------
    assert isinstance(solver, MaxDivSolver)
    assert solver._n == vectors.shape[0]
    assert solver._pdist.shape == ((vectors.shape[0] * (vectors.shape[0] - 1)) // 2,)
    assert solver._k == k
    assert len(solver._solver_steps) == 3
    assert solver._solver_steps[0].name() == init_strategy.name
    assert solver._solver_steps[1].name() == solver_steps[0].name()
    assert solver._solver_steps[2].name() == solver_steps[1].name()
    assert solver._diversity_metric == DiversityMetric.MIN_SEPARATION
    assert solver._constraints == constraints
    assert solver._seed == 123


# =================================================================================================
#  MaxDivSolverBuilder - Constraint penalty
# =================================================================================================
def test_solver_builder_constraint_penalty_default(dummy_problem):
    # --- act ---------------------------------------------
    solver = MaxDivSolverBuilder(dummy_problem).build()

    # --- assert ------------------------------------------
    assert solver._constraint_penalty == ConstraintPenalty.LINEAR


def test_solver_builder_constraint_penalty_quadratic(dummy_problem):
    # --- act ---------------------------------------------
    solver = MaxDivSolverBuilder(dummy_problem).with_constraint_penalty(ConstraintPenalty.QUADRATIC).build()

    # --- assert ------------------------------------------
    assert solver._constraint_penalty == ConstraintPenalty.QUADRATIC


def test_max_div_solver_quadratic_penalty_end_to_end():
    # --- arrange -----------------------------------------
    problem = MaxDivProblem.new(
        vectors=np.random.rand(20, 5).astype(np.float32),
        k=5,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        constraints=[Constraint(set(range(10)), min_count=2, max_count=3)],
    )
    solver = (
        MaxDivSolverBuilder(problem)
        .with_seed(7)
        .with_preset(iterations(100))
        .with_constraint_penalty(ConstraintPenalty.QUADRATIC)
    ).build()

    # --- act ---------------------------------------------
    result = solver.solve(verbosity=0)

    # --- assert ------------------------------------------
    assert len(result.i_selected) == problem.k
    score_after_initialization = result.score_checkpoints[1][2]
    score_after_optimization = result.score_checkpoints[-1][2]
    assert score_after_optimization >= score_after_initialization


# =================================================================================================
#  Presets
# =================================================================================================
@pytest.mark.parametrize("size", [1, 5])
@pytest.mark.parametrize("problem_name", swept_benchmark_problems())
@pytest.mark.parametrize("preset", list(SolverPreset))
def test_max_div_solver_builder_preset(problem_name: str, size: int, preset: SolverPreset):
    """
    Test different preset strategy on reference problems and check if we're optimizing.
    """

    # --- arrange -----------------------------------------

    # prepare problem & solver state
    problem: MaxDivProblem = BenchmarkProblemFactory.construct_problem(
        name=problem_name,
        size=size,
        diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION,
    )

    # --- act ---------------------------------------------
    solver = (
        MaxDivSolverBuilder(problem)
        .with_seed(42)
        .with_preset(
            target_duration=iterations(100),
            preset=preset,
        )
    ).build()
    result = solver.solve(verbosity=25)

    # --- assert ------------------------------------------
    score_after_initialization = result.score_checkpoints[1][2]
    score_after_optimization = result.score_checkpoints[-1][2]

    assert score_after_initialization.size == 1.0, "initialization should select k items."
    assert score_after_optimization.size == 1.0, "final solution should contain k items."
    assert len(result.i_selected) == problem.k, "final solution should contain k items."

    assert score_after_optimization > score_after_initialization, "Optimization should improve the score."
