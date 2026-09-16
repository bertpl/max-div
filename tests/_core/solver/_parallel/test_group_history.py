import numpy as np

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import WorkerGroupChange, WorkerResult, worker_group_history
from max_div._core.solver._solution import MaxDivSolution


def _change(executed_by: int, t: float, n_alive_groups_after: int) -> WorkerGroupChange:
    """Return a change executed by the given worker at its own elapsed `t`, leaving the given group count."""
    return WorkerGroupChange(
        executed_by=executed_by,
        elapsed=Elapsed(t_elapsed_sec=t, n_iterations=int(10 * t)),
        progress_fraction=0.5,
        dissolved_group=n_alive_groups_after,
        n_alive_groups_after=n_alive_groups_after,
        slot_scores={},
        reassignments={n_alive_groups_after: 0},
    )


def _result(worker_index: int, t_start: float, changes: list[WorkerGroupChange]) -> WorkerResult:
    """Return a worker result carrying the given changes and an otherwise empty solution."""
    solution = MaxDivSolution(
        i_selected=np.array([worker_index], dtype=np.int32), score_checkpoints=[], step_durations=[]
    )
    return WorkerResult(
        worker_index=worker_index, seed=worker_index, t_start=t_start, solution=solution, worker_group_changes=changes
    )


def test_changes_are_placed_on_the_shared_axis_and_ordered_by_the_group_count():
    """A later-starting worker's change shifts by its start offset; the order follows the falling group count."""
    # --- arrange ----------------------
    results = [
        _result(0, t_start=100.0, changes=[_change(0, t=5.0, n_alive_groups_after=1)]),
        _result(1, t_start=102.0, changes=[_change(1, t=1.0, n_alive_groups_after=2)]),
    ]

    # --- act --------------------------
    history = worker_group_history(results)

    # --- assert -----------------------
    assert [(c.executed_by, c.elapsed.t_elapsed_sec, c.n_alive_groups_after) for c in history] == [
        (1, 3.0, 2),
        (0, 5.0, 1),
    ]
    assert [c.elapsed.n_iterations for c in history] == [10, 50]  # iteration counts stay the executor's own


def test_the_group_count_orders_changes_whose_clock_readings_disagree():
    """Two changes moments apart keep their true order even when the later executor's reading is earlier."""
    # --- arrange ----------------------
    results = [
        _result(0, t_start=100.0, changes=[_change(0, t=4.0, n_alive_groups_after=2)]),
        _result(1, t_start=100.0, changes=[_change(1, t=3.9, n_alive_groups_after=1)]),
    ]

    # --- act --------------------------
    history = worker_group_history(results)

    # --- assert -----------------------
    assert [c.n_alive_groups_after for c in history] == [2, 1]


def test_workers_without_changes_contribute_nothing():
    """A solve whose workers executed no dissolution has an empty history."""
    # --- arrange ----------------------
    results = [_result(0, t_start=100.0, changes=[]), _result(1, t_start=101.0, changes=[])]

    # --- act / assert -----------------
    assert worker_group_history(results) == []
