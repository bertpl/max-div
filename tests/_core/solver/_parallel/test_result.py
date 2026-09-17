import numpy as np
import pytest

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import WorkerResult
from max_div._core.solver._score import Score
from max_div._core.solver._score_checkpoint import ScoreCheckpoint
from max_div._core.solver._solution import MaxDivSolution
from max_div._core.solver._step_identity import SolverStepIdentity


def _result(worker_index: int, diversity: float, t_start: float = 0.0) -> WorkerResult:
    """Return a result carrying the given diversity and start time, with everything else held equal."""
    score = Score(size=1.0, constraints=1.0, diversities=(diversity,))
    solution = MaxDivSolution(
        i_selected=np.array([worker_index], dtype=np.int32),
        score_checkpoints=[
            ScoreCheckpoint(SolverStepIdentity(1, "step"), Elapsed(t_elapsed_sec=1.0, n_iterations=10), score)
        ],
        step_durations=[],
    )
    return WorkerResult(worker_index=worker_index, seed=worker_index, t_start=t_start, solution=solution)


def test_best_picks_the_highest_score():
    """The winner is the worker that scored best, wherever it sits in the list."""
    # --- arrange / act ----------------
    winner = WorkerResult.best([_result(0, 0.1), _result(1, 0.9), _result(2, 0.5)])

    # --- assert -----------------------
    assert winner.worker_index == 1


def test_best_breaks_ties_by_lowest_worker_index():
    """Equal scores resolve to the lowest index, so the winner never depends on who finished first."""
    # --- arrange ----------------------
    tied = [_result(2, 0.7), _result(0, 0.7), _result(1, 0.7)]

    # --- act --------------------------
    winner = WorkerResult.best(tied)

    # --- assert -----------------------
    assert winner.worker_index == 0
    assert WorkerResult.best(list(reversed(tied))).worker_index == 0  # and not on list order either


def test_best_rejects_an_empty_result_list():
    """No results means every worker failed, which is an error rather than an empty answer."""
    # --- arrange / act / assert -------
    with pytest.raises(ValueError, match="every worker failed"):
        WorkerResult.best([])


# =================================================================================================
#  Placing events on the shared axis
# =================================================================================================
def test_start_offset_is_the_gap_after_the_earliest_worker():
    """A worker's offset is its own start minus the earliest start, and zero for the earliest worker."""
    # --- arrange ----------------------
    results = [_result(0, 0.1, t_start=100.0), _result(1, 0.2, t_start=104.5)]
    t_first_start = WorkerResult.earliest_start_time(results)

    # --- act / assert -----------------
    assert results[0].start_offset_sec(t_first_start) == 0.0
    assert results[1].start_offset_sec(t_first_start) == pytest.approx(4.5)


def test_placing_an_elapsed_shifts_seconds_but_not_iterations():
    """An elapsed counted from the worker's start moves onto the shared axis by the start offset, iterations kept."""
    # --- arrange ----------------------
    worker = _result(1, 0.2, t_start=104.5)

    # --- act --------------------------
    placed = worker.place_on_shared_axis(Elapsed(t_elapsed_sec=2.0, n_iterations=30), t_first_start=100.0)

    # --- assert -----------------------
    assert placed == Elapsed(t_elapsed_sec=6.5, n_iterations=30)
