from dataclasses import replace

import numpy as np
import pytest

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import WorkerResult, best_known_trajectory
from max_div._core.solver._score import Score
from max_div._core.solver._score_checkpoint import ScoreCheckpoint
from max_div._core.solver._solution import MaxDivSolution
from max_div._core.solver._step_identity import SolverStepIdentity


def _result(worker_index: int, t_start: float, trace: list[tuple[float, float]], group_index: int = 0) -> WorkerResult:
    """Return a worker result whose checkpoints hold the given `(elapsed seconds, diversity)` pairs."""
    checkpoints = [
        ScoreCheckpoint(
            SolverStepIdentity(1, "step"),
            Elapsed(t_elapsed_sec=t, n_iterations=int(10 * t)),
            Score(size=1.0, constraints=1.0, diversities=(diversity,)),
            worker_index=worker_index,
            group_index=group_index,
        )
        for t, diversity in trace
    ]
    solution = MaxDivSolution(
        i_selected=np.array([worker_index], dtype=np.int32), score_checkpoints=checkpoints, step_durations={}
    )
    return WorkerResult(worker_index=worker_index, seed=worker_index, t_start=t_start, solution=solution)


def _as_rows(trajectory: list[ScoreCheckpoint]) -> list[tuple[float, float, int]]:
    """Return `(elapsed seconds, diversity, worker)` per checkpoint, for compact assertions."""
    return [(c.elapsed.t_elapsed_sec, c.score.diversity, c.worker_index) for c in trajectory]


def test_the_trajectory_keeps_only_improvements_and_names_their_holder():
    """A checkpoint enters the trace only when it beats the best score so far, tagged with the worker that held it."""
    # --- arrange ----------------------
    results = [
        _result(0, t_start=100.0, trace=[(0.0, 0.1), (1.0, 0.3), (2.0, 0.3)]),
        _result(1, t_start=100.0, trace=[(0.5, 0.2), (1.5, 0.5), (2.0, 0.5)]),
    ]

    # --- act --------------------------
    trajectory = best_known_trajectory(results)

    # --- assert -----------------------
    assert _as_rows(trajectory) == [(0.0, 0.1, 0), (0.5, 0.2, 1), (1.0, 0.3, 0), (1.5, 0.5, 1), (2.0, 0.5, 1)]


def test_a_later_start_shifts_a_worker_s_checkpoints_by_its_offset():
    """Elapsed counts from the earliest worker start, so a worker that started later sits further right."""
    # --- arrange ----------------------
    results = [
        _result(0, t_start=100.0, trace=[(0.0, 0.1), (1.0, 0.1)]),
        _result(1, t_start=100.5, trace=[(0.0, 0.2), (1.0, 0.2)]),
    ]

    # --- act --------------------------
    trajectory = best_known_trajectory(results)

    # --- assert -----------------------
    assert _as_rows(trajectory) == [(0.0, 0.1, 0), (0.5, 0.2, 1), (1.5, 0.2, 1)]


def test_the_trace_closes_at_the_last_checkpoint_holding_the_best_score():
    """The trace ends where the solve ends, so the solution's duration spans the whole parallel solve."""
    # --- arrange ----------------------
    results = [
        _result(0, t_start=0.0, trace=[(0.0, 0.1), (3.0, 0.1)]),  # never improves after the start
        _result(1, t_start=0.0, trace=[(1.0, 0.9), (2.0, 0.9)]),  # reaches the best early, then keeps it
    ]

    # --- act --------------------------
    trajectory = best_known_trajectory(results)

    # --- assert -----------------------
    assert _as_rows(trajectory) == [(0.0, 0.1, 0), (1.0, 0.9, 1), (2.0, 0.9, 1)]
    assert trajectory[-1].elapsed.n_iterations == 20  # the holding worker's own count, not a sum


def test_the_trace_is_the_same_whatever_the_result_order():
    """Ties on time resolve by worker index, so the list order of the results does not change the trace."""
    # --- arrange ----------------------
    results = [_result(0, t_start=0.0, trace=[(0.0, 0.5)]), _result(1, t_start=0.0, trace=[(0.0, 0.5)])]

    # --- act / assert -----------------
    assert _as_rows(best_known_trajectory(results)) == _as_rows(best_known_trajectory(list(reversed(results))))
    assert _as_rows(best_known_trajectory(results)) == [(0.0, 0.5, 0)]


def test_group_tags_survive_the_placement():
    """Placing a checkpoint on the shared axis changes only its elapsed; the group tag stays."""
    # --- arrange / act ----------------
    trajectory = best_known_trajectory([_result(0, t_start=5.0, trace=[(0.0, 0.5)], group_index=3)])

    # --- assert -----------------------
    assert (trajectory[0].worker_index, trajectory[0].group_index) == (0, 3)


def test_a_checkpoint_without_a_worker_is_rejected():
    """A single solve's untagged checkpoint cannot be placed among workers, so the trajectory refuses it."""
    # --- arrange ----------------------
    result = _result(0, t_start=0.0, trace=[(0.0, 0.5)])
    untagged = replace(result.solution.score_checkpoints[0], worker_index=None)
    result.solution.score_checkpoints[0] = untagged

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="name their worker"):
        best_known_trajectory([result])
