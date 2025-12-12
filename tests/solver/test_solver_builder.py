import numpy as np
import pytest

from max_div.solver import Constraint, MaxDivProblem, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._distance import DistanceMetric
from max_div.solver._diversity import DiversityMetric
from max_div.solver._duration import iterations, seconds
from max_div.solver._solver_step import InitializationStep, OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  Fixture
# =================================================================================================
@pytest.fixture
def dummy_problem() -> MaxDivProblem:
    return MaxDivProblem.new(
        vectors=np.random.rand(10, 5).astype(np.float32),
        k=3,
        diversity_metric=DiversityMetric.geomean_separation(),
    )


# =================================================================================================
#  MaxDivSolverBuilder - Modifiers
# =================================================================================================
@pytest.mark.parametrize(
    "strategy, expected_ok",
    [
        (InitializationStrategy.one_shot_random(), True),
        (OptimizationStrategy.dummy(), False),
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
        ([OptimizationStrategy.dummy(), OptimizationStrategy.dummy()], True),
        ([OptimizationStrategy.dummy()], True),
        ([InitializationStrategy.one_shot_random()], False),
        ([InitializationStrategy.one_shot_random(), OptimizationStrategy.dummy()], False),
        ([OptimizationStrategy.dummy(), InitializationStrategy.one_shot_random()], False),
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
            DiversityMetric.min_separation(),
            [DiversityMetric.approx_geomean_separation(), DiversityMetric.non_zero_separation_frac()],
        ),
        (DiversityMetric.geomean_separation(), [DiversityMetric.non_zero_separation_frac()]),
        (DiversityMetric.approx_geomean_separation(), [DiversityMetric.non_zero_separation_frac()]),
        (DiversityMetric.mean_separation(), []),
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
            DiversityMetric.approx_geomean_separation(),
            DiversityMetric.non_zero_separation_frac(),
            DiversityMetric.mean_separation(),
        ]
    )

    # --- act ---------------------------------------------
    solver = builder.build()

    # --- assert ------------------------------------------
    assert solver._diversity_metric == DiversityMetric.geomean_separation()
    assert len(solver._diversity_tie_breakers) == 3
    assert solver._diversity_tie_breakers[0] == DiversityMetric.approx_geomean_separation()
    assert solver._diversity_tie_breakers[1] == DiversityMetric.non_zero_separation_frac()
    assert solver._diversity_tie_breakers[2] == DiversityMetric.mean_separation()


# =================================================================================================
#  MaxDivSolverBuilder - End-to-End
# =================================================================================================
def test_max_div_solver_builder_end_to_end():
    # --- arrange -----------------------------------------
    vectors = np.random.rand(10, 5).astype(np.float32)
    k = 5
    init_strategy = InitializationStrategy.one_shot_random()
    solver_steps = [
        OptimizationStep(OptimizationStrategy.dummy(), seconds(1)),
        OptimizationStep(OptimizationStrategy.dummy(), iterations(100)),
    ]
    constraints = [
        Constraint(set(range(0, 5)), min_count=2, max_count=3),
        Constraint(set(range(5, 10)), min_count=2, max_count=3),
    ]

    # --- act ---------------------------------------------
    builder = (
        MaxDivSolverBuilder(
            MaxDivProblem.new(
                vectors=vectors,
                k=k,
                distance_metric=DistanceMetric.L1_MANHATTAN,
                diversity_metric=DiversityMetric.min_separation(),
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
    assert solver._vectors.shape == vectors.shape
    assert solver._k == k
    assert len(solver._solver_steps) == 3
    assert solver._solver_steps[0].name() == init_strategy.name
    assert solver._solver_steps[1].name() == solver_steps[0].name()
    assert solver._solver_steps[2].name() == solver_steps[1].name()
    assert solver._distance_metric == DistanceMetric.L1_MANHATTAN
    assert solver._diversity_metric.name == DiversityMetric.min_separation().name
    assert solver._constraints == constraints
    assert solver._seed == 123
