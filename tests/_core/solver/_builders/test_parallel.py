import numpy as np
import pytest

from max_div._core._warnings import ParallelSolvingWarning
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder, ParallelMaxDivSolverBuilder
from max_div._core.solver._duration import iterations
from max_div._core.solver._parallel import ParallelMaxDivSolution, WorkerConfig
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._strategies import InitializationStrategy

_BUDGET = iterations(120)


def _problem() -> MaxDivProblem:
    """Return a problem small enough to solve several times over in a test."""
    return MaxDivProblem.new(np.random.default_rng(20260809).random((80, 3)).astype(np.float32), k=8)


def _solve_portfolio(workers, seed: int = 5) -> ParallelMaxDivSolution:
    """Solve the shared test problem with the given workers."""
    return ParallelMaxDivSolverBuilder(_problem()).with_seed(seed).with_workers(_BUDGET, workers).build().solve()


# =================================================================================================
#  Solving
# =================================================================================================
def test_a_portfolio_returns_an_ordinary_solution():
    """The winner is a MaxDivSolution, so code written for a single solve keeps working."""
    # --- arrange / act -----------------------------------
    solution = _solve_portfolio(2)

    # --- assert ------------------------------------------
    assert solution.i_selected.size == 8
    assert solution.score.diversity > 0.0
    assert solution.duration.n_iterations > 0


def test_every_worker_is_summarized():
    """Each worker reports what it ran, what it scored, and whether it reached the best score."""
    # --- arrange / act -----------------------------------
    solution = _solve_portfolio([WorkerConfig(preset=SolverPreset.SMART), WorkerConfig(preset=SolverPreset.GUIDED)])

    # --- assert ------------------------------------------
    assert [worker.worker_index for worker in solution.workers] == [0, 1]
    assert [worker.config.preset for worker in solution.workers] == [SolverPreset.SMART, SolverPreset.GUIDED]
    assert solution.workers[solution.winning_worker].has_best_score


def test_workers_may_differ_by_initialization_alone():
    """Two workers can run one preset from different starting points, which a preset name cannot express."""
    # --- arrange -----------------------------------------
    workers = [
        WorkerConfig(preset=SolverPreset.SMART),
        WorkerConfig(preset=SolverPreset.SMART, init_strategy=InitializationStrategy.farthest_point()),
    ]

    # --- act ---------------------------------------------
    solution = _solve_portfolio(workers)

    # --- assert ------------------------------------------
    assert solution.workers[0].config.init_strategy is None
    assert solution.workers[1].config.init_strategy is not None


def test_workers_at_the_best_score_are_counted():
    """The count says how many workers tied for best; when every worker ties, the portfolio did not help."""
    # --- arrange / act -----------------------------------
    solution = _solve_portfolio(3)

    # --- assert ------------------------------------------
    counted = solution.n_workers_with_best_score
    assert counted == sum(1 for worker in solution.workers if worker.has_best_score)
    assert 1 <= counted <= len(solution.workers)


# =================================================================================================
#  Seeds
# =================================================================================================
def test_one_seed_reproduces_the_whole_portfolio():
    """A portfolio run twice from the same seed selects the same items and seeds its workers alike."""
    # --- arrange / act -----------------------------------
    first, second = _solve_portfolio(2), _solve_portfolio(2)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(first.i_selected, second.i_selected)
    assert [worker.seed for worker in first.workers] == [worker.seed for worker in second.workers]


def test_workers_are_seeded_differently_from_each_other():
    """Derived seeds differ per worker, so the workers search differently."""
    # --- arrange / act -----------------------------------
    seeds = [worker.seed for worker in _solve_portfolio(4).workers]

    # --- assert ------------------------------------------
    assert len(set(seeds)) == len(seeds)


def test_a_worker_can_be_replayed_on_its_own():
    """A summary's preset and seed reproduce that worker's selection in a single solve."""
    # --- arrange -----------------------------------------
    solution = _solve_portfolio([WorkerConfig(preset=SolverPreset.SMART), WorkerConfig(preset=SolverPreset.GUIDED)])
    winner = solution.workers[solution.winning_worker]

    # --- act ---------------------------------------------
    replayed = (
        MaxDivSolverBuilder(_problem())
        .with_preset(_BUDGET, winner.config.preset)
        .with_seed(winner.seed)
        .build()
        .solve(verbosity=0)
    )

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(np.sort(replayed.i_selected), np.sort(solution.i_selected))


# =================================================================================================
#  Guardrails
# =================================================================================================
def test_a_single_worker_warns():
    """One worker cannot beat a single solve, so configuring one worker warns."""
    # --- arrange / act / assert --------------------------
    with pytest.warns(ParallelSolvingWarning, match="cannot do better than solving once"):
        ParallelMaxDivSolverBuilder(_problem()).with_workers(_BUDGET, 1).build()


def test_more_workers_than_cores_warns(monkeypatch: pytest.MonkeyPatch):
    """Configuring more workers than cores warns, because the workers then share cores."""
    # --- arrange -----------------------------------------
    monkeypatch.setattr("max_div._core.solver._parallel._solver.os.cpu_count", lambda: 2)

    # --- act / assert ------------------------------------
    with pytest.warns(ParallelSolvingWarning, match="share cores"):
        ParallelMaxDivSolverBuilder(_problem()).with_workers(_BUDGET, 4).build()


def test_building_without_workers_is_rejected():
    """A portfolio with no workers is a configuration error rather than an empty run."""
    # --- arrange / act / assert --------------------------
    with pytest.raises(ValueError, match="needs workers"):
        ParallelMaxDivSolverBuilder(_problem()).build()
