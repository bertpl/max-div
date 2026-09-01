import warnings

import numpy as np
import pytest

from max_div._core._warnings import SolverBudgetWarning
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import (
    ConstraintPenalty,
    MaxDivSolver,
    MaxDivSolverBuilder,
    SolverPreset,
    Verbosity,
    iterations,
    seconds,
)
from max_div._core.solver._duration import Elapsed
from max_div._core.solver._solver_step import InitializationStep, OptimizationStep, SolverStepResult
from max_div._core.solver._strategies import InitializationStrategy, OptimizationStrategy
from max_div._core.solver._strategies._initialization._init_farthest_point_batched import InitFarthestPointBatched
from max_div._core.solver._strategies._initialization._init_most_feasible import InitMostFeasible
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
    # --- arrange ----------------------
    builder = MaxDivSolverBuilder(dummy_problem)

    # --- act & assert -----------------
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
    # --- arrange ----------------------
    builder = MaxDivSolverBuilder(dummy_problem)
    solver_steps = [
        InitializationStep(s) if isinstance(s, InitializationStrategy) else OptimizationStep(s, duration=seconds(10))
        for s in strategies
    ]

    # --- act & assert -----------------
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
    # --- arrange ----------------------
    builder = MaxDivSolverBuilder(
        MaxDivProblem.new(
            vectors=np.random.rand(10, 5).astype(np.float32),
            k=3,
            diversity_metric=diversity_metric,
        )
    ).with_default_diversity_tie_breakers()

    # --- act --------------------------
    solver = builder.build()

    # --- assert -----------------------
    assert solver._diversity_metric == diversity_metric
    for true_tie_breaker, expected_tie_breaker in zip(solver._diversity_tie_breakers, expected_tie_breakers):
        assert true_tie_breaker == expected_tie_breaker


def test_max_div_solver_builder_tie_breaker_metrics_custom(dummy_problem):
    # --- arrange ----------------------
    builder = MaxDivSolverBuilder(dummy_problem).with_diversity_tie_breakers(
        [
            DiversityMetric.APPROX_GEOMEAN_SEPARATION,
            DiversityMetric.NON_ZERO_SEPARATION_FRAC,
            DiversityMetric.MEAN_SEPARATION,
        ]
    )

    # --- act --------------------------
    solver = builder.build()

    # --- assert -----------------------
    assert solver._diversity_metric == DiversityMetric.GEOMEAN_SEPARATION
    assert len(solver._diversity_tie_breakers) == 3
    assert solver._diversity_tie_breakers[0] == DiversityMetric.APPROX_GEOMEAN_SEPARATION
    assert solver._diversity_tie_breakers[1] == DiversityMetric.NON_ZERO_SEPARATION_FRAC
    assert solver._diversity_tie_breakers[2] == DiversityMetric.MEAN_SEPARATION


# =================================================================================================
#  MaxDivSolverBuilder - Store built in solve(), not build()
# =================================================================================================
def test_the_store_is_built_by_solve_not_by_build(dummy_problem, monkeypatch):
    """build() only assembles the solver; each solve() builds the store, so its cost sits in solve()."""
    # --- arrange ----------------------
    from max_div._core.solver._builders import _single

    builds = 0
    real_build = _single.build_distance_store

    def counting_build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(_single, "build_distance_store", counting_build)

    # --- act / assert -----------------
    solver = MaxDivSolverBuilder(dummy_problem).with_preset(iterations(5), SolverPreset.RANDOM).build()
    assert builds == 0  # build() did not touch the distances

    solver.solve(verbosity=Verbosity.SILENT)
    solver.solve(verbosity=Verbosity.SILENT)
    assert builds == 2  # one store built per solve


# =================================================================================================
#  MaxDivSolverBuilder - End-to-End
# =================================================================================================
def test_max_div_solver_builder_end_to_end():
    # --- arrange ----------------------
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

    # --- act --------------------------
    builder = (
        MaxDivSolverBuilder(
            MaxDivProblem.new(
                vectors=vectors,
                k=k,
                distance_metric=DistanceMetric.l1_manhattan(),
                diversity_metric=DiversityMetric.MIN_SEPARATION,
                constraints=constraints,
            )
        )
        .set_initialization_strategy(init_strategy)
        .add_solver_steps(solver_steps)
        .with_seed(123)
    )
    solver = builder.build()

    # --- assert -----------------------
    assert isinstance(solver, MaxDivSolver)
    assert solver._n == vectors.shape[0]
    # the store is built by solve(), not build(): a full-matrix store of the expected shape here
    store = solver._store_provider()
    assert store.matrix.shape == (vectors.shape[0], vectors.shape[0])  # AUTO -> full matrix at this size
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
    # --- act --------------------------
    solver = MaxDivSolverBuilder(dummy_problem).build()

    # --- assert -----------------------
    assert solver._constraint_penalty == ConstraintPenalty.LINEAR


def test_solver_builder_constraint_penalty_quadratic(dummy_problem):
    # --- act --------------------------
    solver = MaxDivSolverBuilder(dummy_problem).with_constraint_penalty(ConstraintPenalty.QUADRATIC).build()

    # --- assert -----------------------
    assert solver._constraint_penalty == ConstraintPenalty.QUADRATIC


def test_max_div_solver_quadratic_penalty_end_to_end():
    # --- arrange ----------------------
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

    # --- act --------------------------
    result = solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert len(result.i_selected) == problem.k
    score_after_initialization = result.score_checkpoints[1][2]
    score_after_optimization = result.score_checkpoints[-1][2]
    assert score_after_optimization >= score_after_initialization


# =================================================================================================
#  Presets
# =================================================================================================
@pytest.mark.parametrize("n", [100, 500])
@pytest.mark.parametrize("problem_name", swept_benchmark_problems())
@pytest.mark.parametrize("preset", list(SolverPreset))
def test_max_div_solver_builder_preset(problem_name: str, n: int, preset: SolverPreset):
    """
    Test different preset strategy on reference problems and check if we're optimizing.
    """

    # --- arrange ----------------------

    # prepare problem & solver state
    problem: MaxDivProblem = BenchmarkProblemFactory.construct_problem(
        name=problem_name,
        n=n,
        diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION,
    )

    # --- act --------------------------
    solver = (
        MaxDivSolverBuilder(problem)
        .with_seed(42)
        .with_preset(
            target_duration=iterations(100),
            preset=preset,
        )
    ).build()
    result = solver.solve(verbosity=Verbosity.TABULAR_DEBUG)

    # --- assert -----------------------
    score_after_initialization = result.score_checkpoints[1][2]
    score_after_optimization = result.score_checkpoints[-1][2]

    assert score_after_initialization.size == 1.0, "initialization should select k items."
    assert score_after_optimization.size == 1.0, "final solution should contain k items."
    assert len(result.i_selected) == problem.k, "final solution should contain k items."

    assert score_after_optimization > score_after_initialization, "Optimization should improve the score."


@pytest.mark.parametrize("preset", [SolverPreset.SMART, SolverPreset.THOROUGH])
@pytest.mark.parametrize(
    "constraints,expected_init",
    [
        ([], InitFarthestPointBatched),
        ([Constraint(set(range(10)), min_count=2, max_count=3)], InitMostFeasible),
    ],
)
def test_with_preset_switches_init_on_constraints(
    preset: SolverPreset, constraints: list[Constraint], expected_init: type[InitializationStrategy]
):
    """with_preset gives SMART/THOROUGH the feasibility-witness init when the problem is constrained."""
    # --- arrange ----------------------
    problem = MaxDivProblem.new(
        vectors=np.random.rand(20, 5).astype(np.float32),
        k=5,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        constraints=constraints,
    )

    # --- act --------------------------
    builder = MaxDivSolverBuilder(problem).with_preset(iterations(100), preset)

    # --- assert -----------------------
    assert isinstance(builder._solver_steps[0]._strategy, expected_init)


# =================================================================================================
#  MaxDivSolverBuilder - end-to-end budget
# =================================================================================================
def test_an_end_to_end_budget_requires_a_time_budget(dummy_problem):
    """An iteration count cannot bound the store build and initialization, so build() rejects the combination."""
    # --- arrange / act / assert -------
    with pytest.raises(ValueError, match="requires a time budget"):
        MaxDivSolverBuilder(dummy_problem).with_preset(iterations(100)).with_end_to_end_budget().build()


def test_a_budget_spent_during_setup_skips_the_optimization(dummy_problem, fake_clock):
    """With the budget gone before optimization starts, the solve warns and returns the initialization's selection."""
    # --- arrange ----------------------
    solver = (
        MaxDivSolverBuilder(dummy_problem)
        .with_preset(seconds(10.0), SolverPreset.RANDOM)
        .with_end_to_end_budget()
        .build()
    )
    store_provider = solver._store_provider
    solver._store_provider = lambda: (fake_clock.advance(11.0), store_provider())[1]  # the build eats the budget

    # --- act --------------------------
    with pytest.warns(SolverBudgetWarning, match="spent before optimization started"):
        solution = solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    optimization_steps = [name for name in solution.step_durations if "OptimRandomSwaps" in name]
    assert len(optimization_steps) == 1
    assert solution.step_durations[optimization_steps[0]].n_iterations == 0
    assert len(solution.i_selected) == dummy_problem.k


def test_solve_hands_every_step_the_budget_and_its_start(dummy_problem, fake_clock, monkeypatch):
    """Each solve pushes the budget and the solve-start time into the steps, like the seeds."""
    # --- arrange ----------------------
    solver = (
        MaxDivSolverBuilder(dummy_problem)
        .with_preset(seconds(10.0), SolverPreset.RANDOM)
        .with_end_to_end_budget()
        .build()
    )
    fake_clock.advance(4.0)  # time before solve is not part of the budget
    received = []
    original = OptimizationStep.set_e2e_budget

    def record_budget(self, e2e_budget):
        received.append((e2e_budget.budget_sec, e2e_budget.t_start))
        original(self, e2e_budget)

    monkeypatch.setattr(OptimizationStep, "set_e2e_budget", record_budget)
    # stub the run: under a frozen fake clock a real time-budgeted run would never finish
    monkeypatch.setattr(
        OptimizationStep,
        "run",
        lambda self, state, *args, **kwargs: SolverStepResult(
            score_checkpoints=[(Elapsed(t_elapsed_sec=0.0, n_iterations=0), state.score)]
        ),
    )

    # --- act --------------------------
    solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert received == [(10.0, fake_clock.monotonic())]


@pytest.mark.parametrize("duration", [seconds(0.05), iterations(5)])
def test_a_step_budget_never_warns_about_a_spent_budget(dummy_problem, duration):
    """Only an end-to-end budget can arrive spent; the warning must not fire without the flag."""
    # --- arrange ----------------------
    solver = MaxDivSolverBuilder(dummy_problem).with_preset(duration, SolverPreset.RANDOM).build()

    # --- act --------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert [w for w in caught if issubclass(w.category, SolverBudgetWarning)] == []
