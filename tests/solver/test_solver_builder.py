import numpy as np
import pytest

from max_div.solver import Constraint, MaxDivSolver, MaxDivSolverBuilder
from max_div.solver._distance import DistanceMetric
from max_div.solver._diversity import DiversityMetric
from max_div.solver._strategies import SolverStrategy


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
        (SolverStrategy.init_random(), True),
        (SolverStrategy.optim_dummy(), False),
    ],
)
def test_solver_builder_set_initialization_strategy(strategy: SolverStrategy, expected_ok: bool):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder()

    # --- act & assert ------------------------------------
    if expected_ok:
        builder = builder.set_initialization_strategy(strategy)
        assert builder is not None
    else:
        with pytest.raises(ValueError):
            _ = builder.set_initialization_strategy(strategy)


@pytest.mark.parametrize(
    "strategies, expected_ok",
    [
        ([SolverStrategy.optim_dummy(), SolverStrategy.optim_dummy()], True),
        ([SolverStrategy.optim_dummy()], True),
        ([SolverStrategy.init_random()], False),
        ([SolverStrategy.init_random(), SolverStrategy.optim_dummy()], False),
        ([SolverStrategy.optim_dummy(), SolverStrategy.init_random()], False),
    ],
)
def test_solver_builder_add_optimization_strategies(strategies: list[SolverStrategy], expected_ok: bool):
    # --- arrange -----------------------------------------
    builder = MaxDivSolverBuilder()

    # --- act & assert ------------------------------------
    if expected_ok:
        builder = builder.add_optimization_strategies(strategies)
        assert builder is not None
    else:
        with pytest.raises(ValueError):
            _ = builder.add_optimization_strategies(strategies)


# =================================================================================================
#  MaxDivSolverBuilder - End-to-End
# =================================================================================================
def test_max_div_solver_builder_end_to_end():
    # --- arrange -----------------------------------------
    vectors = np.random.rand(10, 5).astype(np.float32)
    selection_size = 5
    init_strategy = SolverStrategy.init_random()
    optim_strategies = [SolverStrategy.optim_dummy(), SolverStrategy.optim_dummy()]
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
        .add_optimization_strategies(optim_strategies)
        .with_distance_metric(DistanceMetric.L1_MANHATTAN)
        .with_diversity_metric(DiversityMetric.min_separation())
        .with_constraints(constraints)
    )
    solver = builder.build()

    # --- assert ------------------------------------------
    assert isinstance(solver, MaxDivSolver)
    assert solver._vectors.shape == vectors.shape
    assert solver._selection_size == selection_size
    assert solver._strategies == [init_strategy] + optim_strategies
    assert solver._distance_metric == DistanceMetric.L1_MANHATTAN
    assert solver._diversity_metric.name == DiversityMetric.min_separation().name
    assert solver._constraints == constraints
