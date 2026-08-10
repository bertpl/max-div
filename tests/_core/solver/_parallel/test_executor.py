import numpy as np
import pytest

from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder
from max_div._core.solver._distance_storage import build_shared_distance_store
from max_div._core.solver._duration import iterations
from max_div._core.solver._parallel import IndependentCoordinator, best_result, run_portfolio
from max_div._core.solver._presets import SolverPreset

_SEEDS = (11, 22, 33)


def _builder() -> MaxDivSolverBuilder:
    """Return a builder over a problem small enough to solve several times in a test."""
    rng = np.random.default_rng(20260809)
    problem = MaxDivProblem.new(rng.random((60, 3)).astype(np.float32), k=6)
    return MaxDivSolverBuilder(problem).with_preset(iterations(50), SolverPreset.SMART)


@pytest.fixture
def portfolio_results():
    """Run one portfolio over spawned workers, and hand back its results with the builder used."""
    builder = _builder()
    resolved, config = builder.prepare_storage_and_config()
    with build_shared_distance_store(builder._problem, resolved) as shared:
        yield builder, run_portfolio([config.with_seed(seed) for seed in _SEEDS], shared.spec, IndependentCoordinator())


def test_every_worker_reports_a_result(portfolio_results):
    """A portfolio returns one result per configuration, each carrying a full selection."""
    # --- arrange / act -----------------------------------
    builder, results = portfolio_results

    # --- assert ------------------------------------------
    assert len(results) == len(_SEEDS)
    assert all(result.i_selected.size == builder._k for result in results)


def test_results_come_back_in_worker_order(portfolio_results):
    """Results are ordered by worker rather than by who finished first."""
    # --- arrange / act -----------------------------------
    _, results = portfolio_results

    # --- assert ------------------------------------------
    assert [result.worker_index for result in results] == list(range(len(_SEEDS)))
    assert [result.seed for result in results] == list(_SEEDS)


def test_a_worker_reproduces_the_same_solve_run_alone(portfolio_results):
    """A worker's selection is what the same configuration and seed produce in a single process."""
    # --- arrange -----------------------------------------
    builder, results = portfolio_results

    # --- act ---------------------------------------------
    alone = builder.with_seed(_SEEDS[0]).build().solve(verbosity=0)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(np.sort(results[0].i_selected), np.sort(alone.i_selected))


def test_the_best_reported_result_wins(portfolio_results):
    """The winner is the best-scoring worker, not the first to report."""
    # --- arrange / act -----------------------------------
    _, results = portfolio_results
    winner = best_result(results)

    # --- assert ------------------------------------------
    assert winner.score == max(result.score for result in results)
