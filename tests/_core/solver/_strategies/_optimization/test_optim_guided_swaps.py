from typing import TYPE_CHECKING, Any

import pytest

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._parameters import ParameterSchedule, linear
from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._strategies._initialization import InitializationStrategy
from max_div._core.solver._strategies._optimization import OptimizationStrategy
from max_div._core.solver._strategies._optimization._optim_guided_swaps import OptimGuidedSwaps
from tests.helpers import swept_benchmark_problems

if TYPE_CHECKING:
    from max_div._core.problem import MaxDivProblem


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
@pytest.mark.parametrize("problem_name", swept_benchmark_problems())
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
        diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION,
    )
    solver_state = SolverState.new(
        n=problem.n,
        store=DistanceStore.condensed(problem.condensed_distances(), n=problem.n),
        k=problem.k,
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
    optim_strategy.set_seed(42)
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
        ({"min_swap_size": 1, "max_swap_size": 1}, "OptimGuidedSwaps(1)"),
        ({"min_swap_size": 2, "max_swap_size": 2}, "OptimGuidedSwaps(2)"),
        ({"min_swap_size": 3, "max_swap_size": 3}, "OptimGuidedSwaps(3)"),
        ({"min_swap_size": 1, "max_swap_size": 3}, "OptimGuidedSwaps(1-3)"),
        ({"min_swap_size": 2, "max_swap_size": 5}, "OptimGuidedSwaps(2-5)"),
    ],
)
def test_optim_guided_swaps_name(kwargs: dict[str, Any], expected_name: str):
    """Test that the strategy name is generated as expected."""

    # --- arrange -----------------------------------------
    optim_strategy = OptimizationStrategy.guided_swaps(**kwargs)

    # --- act & assert ------------------------------------
    assert optim_strategy.name == expected_name


def test_optim_guided_swaps_get_debug_info():
    # --- arrange -----------------------------------------
    strategy = OptimGuidedSwaps(
        min_swap_size=2,
        max_swap_size=5,
        swap_size_lambda=2.1,
        add_selectivity_modifier=0.6,
        remove_selectivity_modifier=-0.2,
        p_add_constraint_aware=0.4,
        constraint_softness=0.23,
    )

    # --- act ---------------------------------------------
    debug_info = strategy.get_debug_info()
    # --- assert ------------------------------------------
    assert (
        debug_info.strip() == "scs= 50.000% | λ_swap= 2.10 | sel_rem=-0.20 | sel_add= 0.60 | p_con= 0.40 | soft= 0.23"
    )
