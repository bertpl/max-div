from typing import Any

import pytest

from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import DiversityMetric, MaxDivProblem
from max_div.solver._scheduling import ParameterSchedule, linear
from max_div.solver._solver_state import SolverState
from max_div.solver._solver_step import InitializationStep
from max_div.solver._strategies._initialization import InitializationStrategy
from max_div.solver._strategies._optimization import OptimizationStrategy


@pytest.mark.parametrize(
    "min_swap_size, max_swap_size, swap_size_lambda",
    [
        (1, 1, 1.0),
        (1, 3, linear(0.5, 3.5)),
    ],
)
@pytest.mark.parametrize(
    "add_selectivity_modifier, remove_selectivity_modifier",
    [
        (0.0, 0.0),
        (linear(0.5, -0.5), linear(0.5, -0.5)),
    ],
)
@pytest.mark.parametrize(
    "p_add_constraint_aware",
    [
        0.1,
        0.9,
        linear(0.0, 1.0),
    ],
)
@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("problem_name", ["A1", "A2", "A3", "A4", "A5"])
def test_optim_guided_swaps(
    problem_name: str,
    size: int,
    min_swap_size: int,
    max_swap_size: int,
    swap_size_lambda: float | ParameterSchedule,
    add_selectivity_modifier: float | ParameterSchedule,
    remove_selectivity_modifier: float | ParameterSchedule,
    p_add_constraint_aware: float | ParameterSchedule,
):
    """
    Test OptimGuidedSwaps strategy on reference problems, with very rudimentary initialization,
    to see if we're optimizing.
    """

    # --- arrange -----------------------------------------

    # prepare problem & solver state
    problem: MaxDivProblem = BenchmarkProblemFactory.construct_problem(
        name=problem_name,
        size=size,
        diversity_metric=DiversityMetric.approx_geomean_separation(),
    )
    solver_state = SolverState.new(
        vectors=problem.vectors,
        k=problem.k,
        distance_metric=problem.distance_metric,
        diversity_metric=problem.diversity_metric,
        diversity_tie_breakers=[],
        constraints=problem.constraints,
    )

    # initialize solver state
    init_step = InitializationStep(InitializationStrategy.fast())
    init_step.run(solver_state)

    # prepare strategy
    optim_strategy = OptimizationStrategy.guided_swaps(
        min_swap_size=min_swap_size,
        max_swap_size=max_swap_size,
        swap_size_lambda=swap_size_lambda,
    )
    optim_strategy.seed = 42
    initial_score = solver_state.score
    n_iterations = 100

    # --- act ---------------------------------------------
    optim_strategy.perform_n_iterations(
        state=solver_state,
        n_iters=n_iterations,
        current_progress_frac=0.0,
        progress_frac_per_iter=1e-6,  # dummy value
    )

    # --- assert ------------------------------------------
    assert len(solver_state.selected_index_array) == problem.k, "Number of selected items should remain k."
    assert solver_state.score > initial_score, "We should be optimizing"


@pytest.mark.parametrize(
    "kwargs, expected_name",
    [
        (dict(min_swap_size=1, max_swap_size=1), "OptimGuidedSwaps(1)"),
        (dict(min_swap_size=2, max_swap_size=2), "OptimGuidedSwaps(2)"),
        (dict(min_swap_size=3, max_swap_size=3), "OptimGuidedSwaps(3)"),
        (dict(min_swap_size=1, max_swap_size=3), "OptimGuidedSwaps(1-3)"),
        (dict(min_swap_size=2, max_swap_size=5), "OptimGuidedSwaps(2-5)"),
    ],
)
def test_optim_guided_swaps_name(kwargs: dict[str, Any], expected_name: str):
    """Test that the strategy name is generated as expected."""

    # --- arrange -----------------------------------------
    optim_strategy = OptimizationStrategy.guided_swaps(**kwargs)

    # --- act & assert ------------------------------------
    assert optim_strategy.name == expected_name
