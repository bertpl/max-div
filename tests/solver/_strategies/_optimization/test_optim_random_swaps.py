import pytest

from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import DiversityMetric, MaxDivProblem
from max_div.solver._solver_state import SolverState
from max_div.solver._solver_step import InitializationStep
from max_div.solver._strategies._initialization import InitializationStrategy
from max_div.solver._strategies._optimization import OptimizationStrategy


@pytest.mark.parametrize("size", [1, 2, 10])
@pytest.mark.parametrize("problem_name", BenchmarkProblemFactory.get_all_benchmark_names())
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
