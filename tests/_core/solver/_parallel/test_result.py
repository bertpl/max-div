import numpy as np
import pytest

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import WorkerResult, best_result
from max_div._core.solver._score import Score


def _result(worker_index: int, diversity: float) -> WorkerResult:
    """Return a result carrying the given diversity, with everything else held equal."""
    return WorkerResult(
        worker_index=worker_index,
        i_selected=np.array([worker_index], dtype=np.int32),
        score=Score(size=1.0, constraints=1.0, diversity=diversity, div_tie_breakers=()),
        elapsed=Elapsed(t_elapsed_sec=1.0, n_iterations=10),
        seed=worker_index,
    )


def test_best_result_picks_the_highest_score():
    """The winner is the worker that scored best, wherever it sits in the list."""
    # --- arrange / act -----------------------------------
    winner = best_result([_result(0, 0.1), _result(1, 0.9), _result(2, 0.5)])

    # --- assert ------------------------------------------
    assert winner.worker_index == 1


def test_best_result_breaks_ties_by_lowest_worker_index():
    """Equal scores resolve to the lowest index, so the winner never depends on who finished first."""
    # --- arrange -----------------------------------------
    tied = [_result(2, 0.7), _result(0, 0.7), _result(1, 0.7)]

    # --- act ---------------------------------------------
    winner = best_result(tied)

    # --- assert ------------------------------------------
    assert winner.worker_index == 0
    assert best_result(list(reversed(tied))).worker_index == 0  # and not on list order either


def test_best_result_rejects_an_empty_portfolio():
    """No results means every worker failed, which is an error rather than an empty answer."""
    # --- arrange / act / assert --------------------------
    with pytest.raises(ValueError, match="every worker failed"):
        best_result([])
