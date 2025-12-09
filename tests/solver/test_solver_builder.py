import numpy as np
import pytest

from max_div.solver import Constraint, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._distance import DistanceMetric
from max_div.solver._diversity import DiversityMetric
from max_div.solver._duration import iterations, seconds
from max_div.solver._solver_step import InitializationStep, OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  MaxDivSolverBuilder - Build Validation
# =================================================================================================
@pytest.mark.parametrize(
    "selection_size, set_vectors, expected_buildable",
    [
        (5, True, True),  # Valid case: vectors set and selection size valid
        (2, True, True),  # Valid case: minimum valid selection size
        (10, True, True),  # Valid case: selection size equal to number of vectors
        (11, True, False),  # Invalid case: selection size greater than number of vectors
        (5, False, False),  # Invalid case: vectors not set
        (None, True, False),  # Invalid case: selection size not set
        (None, False, False),  # Invalid case: neither vectors nor selection size set
    ],
)
def test_solver_builder_is_buildable(selection_size: int | None, set_vectors: bool, expected_buildable: bool):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder()
    if set_vectors:
        vectors = np.random.rand(10, 5).astype(np.float32)
        builder = builder.with_vectors(vectors)
    if selection_size is not None:
        builder = builder.with_selection_size(selection_size)

    # --- act & assert ------------------------------------
    if expected_buildable:
        solver = builder.build()
        assert isinstance(solver, MaxDivSolver)
    else:
        with pytest.raises(ValueError):
            _ = builder.build()


# =================================================================================================
#  MaxDivSolverBuilder - Modifiers
# =================================================================================================
@pytest.mark.parametrize(
    "vectors, expected_ok",
    [
        (np.random.rand(10, 5).astype(np.float32), True),  # Valid case: correct dtype & valid dimensions
        (np.random.rand(10, 5).astype(np.float64), False),  # Invalid case: wrong dtype
        (np.random.rand(10).astype(np.float32), False),  # Invalid case: 1D array
        (np.random.rand(1, 5).astype(np.float32), False),  # Invalid case: less than 2 vectors
        (np.random.rand(10, 0).astype(np.float32), False),  # Invalid case: 0 dimensions
    ],
)
def test_solver_builder_with_vectors(vectors: np.ndarray, expected_ok: bool):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder()

    # --- act & assert ------------------------------------
    if expected_ok:
        builder = builder.with_vectors(vectors)
        assert builder is not None
    else:
        with pytest.raises(ValueError):
            _ = builder.with_vectors(vectors)


def test_solver_builder_with_selection_size():
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder()

    # --- act & assert ------------------------------------
    # Valid case
    builder = builder.with_selection_size(5)
    assert builder is not None

    # Invalid case: selection size less than 2
    with pytest.raises(ValueError):
        _ = builder.with_selection_size(1)


@pytest.mark.parametrize(
    "strategy, expected_ok",
    [
        (InitializationStrategy.random(), True),
        (OptimizationStrategy.dummy(), False),
    ],
)
def test_solver_builder_set_initialization_strategy(strategy, expected_ok: bool):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder()

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
        ([InitializationStrategy.random()], False),
        ([InitializationStrategy.random(), OptimizationStrategy.dummy()], False),
        ([OptimizationStrategy.dummy(), InitializationStrategy.random()], False),
    ],
)
def test_solver_builder_add_solver_steps(strategies: list, expected_ok: bool):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder()
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
    builder = (
        MaxDivSolverBuilder()
        .with_vectors(np.random.rand(10, 5).astype(np.float32))
        .with_selection_size(4)
        .with_diversity_metric(diversity_metric)
        .with_default_diversity_tie_breakers()
    )

    # --- act ---------------------------------------------
    solver = builder.build()

    # --- assert ------------------------------------------
    assert solver._diversity_metric == diversity_metric
    for true_tie_breaker, expected_tie_breaker in zip(solver._diversity_tie_breakers, expected_tie_breakers):
        assert true_tie_breaker == expected_tie_breaker


def test_max_div_solver_builder_tie_breaker_metrics_custom():
    # --- arrange -----------------------------------------
    builder = (
        MaxDivSolverBuilder()
        .with_vectors(np.random.rand(10, 5).astype(np.float32))
        .with_selection_size(4)
        .with_diversity_metric(DiversityMetric.geomean_separation())
        .with_diversity_tie_breakers(
            [
                DiversityMetric.approx_geomean_separation(),
                DiversityMetric.non_zero_separation_frac(),
                DiversityMetric.mean_separation(),
            ]
        )
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
    selection_size = 5
    init_strategy = InitializationStrategy.random()
    solver_steps = [
        OptimizationStep(OptimizationStrategy.dummy(), seconds(1)),
        OptimizationStep(OptimizationStrategy.dummy(), iterations(100)),
    ]
    constraints = [
        Constraint(set(range(0, 5)), min_count=2, max_count=3),
        Constraint(set(range(5, 10)), min_count=2, max_count=3),
    ]

    builder = MaxDivSolverBuilder()

    # --- act ---------------------------------------------
    builder = (
        builder.with_vectors(vectors)
        .with_selection_size(selection_size)
        .set_initialization_strategy(init_strategy)
        .add_solver_steps(solver_steps)
        .with_distance_metric(DistanceMetric.L1_MANHATTAN)
        .with_diversity_metric(DiversityMetric.min_separation())
        .with_constraints(constraints)
    )
    solver = builder.build()

    # --- assert ------------------------------------------
    assert isinstance(solver, MaxDivSolver)
    assert solver._vectors.shape == vectors.shape
    assert solver._selection_size == selection_size
    assert len(solver._solver_steps) == 3
    assert solver._solver_steps[0].name() == init_strategy.name
    assert solver._solver_steps[1].name() == solver_steps[0].name()
    assert solver._solver_steps[2].name() == solver_steps[1].name()
    assert solver._distance_metric == DistanceMetric.L1_MANHATTAN
    assert solver._diversity_metric.name == DiversityMetric.min_separation().name
    assert solver._constraints == constraints
