from typing import TYPE_CHECKING

import pytest

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.metrics._distance import condensed_store
from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._strategies._initialization import InitializationStrategy
from max_div._core.solver._strategies._optimization import OptimizationStrategy
from tests.helpers import swept_benchmark_problems

if TYPE_CHECKING:
    from max_div._core.problem import MaxDivProblem


@pytest.mark.parametrize("size", [1, 2, 10])
@pytest.mark.parametrize("problem_name", swept_benchmark_problems())
def test_optim_random_swaps(problem_name: str, size: int):
    """
    Test OptimRandomSwaps strategy on reference problems, with very rudimentary initialization,
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
        store=condensed_store(problem.condensed_distances(), n=problem.n),
        k=problem.k,
        diversity_metric=problem.diversity_metric,
        diversity_tie_breakers=[],
        constraints=problem.constraints,
    )

    # initialize solver state
    init_step = InitializationStep(InitializationStrategy.fast())
    init_step.run(solver_state)

    # prepare strategy
    optim_strategy = OptimizationStrategy.random_swaps()
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
