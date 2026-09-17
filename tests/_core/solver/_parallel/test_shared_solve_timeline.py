import numpy as np

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import SharedSolveTimeline, WorkerGroupChange, WorkerResult
from max_div._core.solver._score import Score
from max_div._core.solver._score_checkpoint import ScoreCheckpoint
from max_div._core.solver._solution import MaxDivSolution
from max_div._core.solver._step_identity import SolverStepIdentity


def _result(
    worker_index: int,
    t_start: float,
    trace: list[tuple[float, float]] | None = None,
    changes: list[WorkerGroupChange] | None = None,
    group_index: int = 0,
) -> WorkerResult:
    """Return a worker result whose checkpoints hold the given `(elapsed seconds, diversity)` pairs, plus its changes."""
    checkpoints = [
        ScoreCheckpoint(
            SolverStepIdentity(1, "step"),
            Elapsed(t_elapsed_sec=t, n_iterations=int(10 * t)),
            Score(size=1.0, constraints=1.0, diversities=(diversity,)),
            worker_index=worker_index,
            group_index=group_index,
        )
        for t, diversity in (trace or [])
    ]
    solution = MaxDivSolution(
        i_selected=np.array([worker_index], dtype=np.int32), score_checkpoints=checkpoints, step_durations=[]
    )
    return WorkerResult(
        worker_index=worker_index,
        seed=worker_index,
        t_start=t_start,
        solution=solution,
        worker_group_changes=changes or [],
    )


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


def _checkpoint_rows(checkpoints: list[ScoreCheckpoint]) -> list[tuple[float, float, int]]:
    """Return `(elapsed seconds, diversity, worker)` per checkpoint, for compact assertions."""
    return [(c.elapsed.t_elapsed_sec, c.score.diversity, c.worker_index) for c in checkpoints]


# =================================================================================================
#  Aligning onto the shared axis
# =================================================================================================
def test_start_offsets_are_measured_from_the_earliest_worker():
    """Each offset is a worker's start minus the earliest start, and zero for the earliest worker."""
    # --- arrange / act ----------------
    timeline = SharedSolveTimeline.from_worker_results(
        [_result(0, t_start=104.5), _result(1, t_start=100.0), _result(2, t_start=106.0)]
    )

    # --- assert -----------------------
    assert timeline.start_offsets == {0: 4.5, 1: 0.0, 2: 6.0}


# =================================================================================================
#  Best-known checkpoints
# =================================================================================================
def test_the_trace_keeps_only_improvements_and_names_their_holder():
    """A checkpoint enters the trace only when it beats the best score so far, tagged with the worker that held it."""
    # --- arrange ----------------------
    timeline = SharedSolveTimeline.from_worker_results(
        [
            _result(0, t_start=100.0, trace=[(0.0, 0.1), (1.0, 0.3), (2.0, 0.3)]),
            _result(1, t_start=100.0, trace=[(0.5, 0.2), (1.5, 0.5), (2.0, 0.5)]),
        ]
    )

    # --- act --------------------------
    trace = timeline.best_known_checkpoints()

    # --- assert -----------------------
    assert _checkpoint_rows(trace) == [(0.0, 0.1, 0), (0.5, 0.2, 1), (1.0, 0.3, 0), (1.5, 0.5, 1), (2.0, 0.5, 1)]


def test_a_later_start_shifts_a_worker_s_checkpoints_by_its_offset():
    """Elapsed counts from the earliest worker start, so a worker that started later sits further right."""
    # --- arrange ----------------------
    timeline = SharedSolveTimeline.from_worker_results(
        [
            _result(0, t_start=100.0, trace=[(0.0, 0.1), (1.0, 0.1)]),
            _result(1, t_start=100.5, trace=[(0.0, 0.2), (1.0, 0.2)]),
        ]
    )

    # --- act --------------------------
    trace = timeline.best_known_checkpoints()

    # --- assert -----------------------
    assert _checkpoint_rows(trace) == [(0.0, 0.1, 0), (0.5, 0.2, 1), (1.5, 0.2, 1)]


def test_the_trace_closes_at_the_last_checkpoint_holding_the_best_score():
    """The trace ends where the solve ends, so the solution's duration spans the whole parallel solve."""
    # --- arrange ----------------------
    timeline = SharedSolveTimeline.from_worker_results(
        [
            _result(0, t_start=0.0, trace=[(0.0, 0.1), (3.0, 0.1)]),  # never improves after the start
            _result(1, t_start=0.0, trace=[(1.0, 0.9), (2.0, 0.9)]),  # reaches the best early, then keeps it
        ]
    )

    # --- act --------------------------
    trace = timeline.best_known_checkpoints()

    # --- assert -----------------------
    assert _checkpoint_rows(trace) == [(0.0, 0.1, 0), (1.0, 0.9, 1), (2.0, 0.9, 1)]
    assert trace[-1].elapsed.n_iterations == 20  # the holding worker's own count, not a sum


def test_the_trace_is_the_same_whatever_the_result_order():
    """Ties on time resolve by worker index, so the list order of the results does not change the trace."""
    # --- arrange ----------------------
    results = [_result(0, t_start=0.0, trace=[(0.0, 0.5)]), _result(1, t_start=0.0, trace=[(0.0, 0.5)])]

    # --- act --------------------------
    forward = SharedSolveTimeline.from_worker_results(results).best_known_checkpoints()
    reversed_ = SharedSolveTimeline.from_worker_results(list(reversed(results))).best_known_checkpoints()

    # --- assert -----------------------
    assert _checkpoint_rows(forward) == _checkpoint_rows(reversed_) == [(0.0, 0.5, 0)]


def test_group_tags_survive_the_placement():
    """Placing a checkpoint on the shared axis changes only its elapsed; the group tag stays."""
    # --- arrange / act ----------------
    timeline = SharedSolveTimeline.from_worker_results([_result(0, t_start=5.0, trace=[(0.0, 0.5)], group_index=3)])
    trace = timeline.best_known_checkpoints()

    # --- assert -----------------------
    assert (trace[0].worker_index, trace[0].group_index) == (0, 3)


# =================================================================================================
#  Group history
# =================================================================================================
def test_changes_are_placed_on_the_shared_axis_and_ordered_by_the_group_count():
    """A later-starting worker's change shifts by its start offset; the order follows the falling group count."""
    # --- arrange ----------------------
    timeline = SharedSolveTimeline.from_worker_results(
        [
            _result(0, t_start=100.0, changes=[_change(0, t=5.0, n_alive_groups_after=1)]),
            _result(1, t_start=102.0, changes=[_change(1, t=1.0, n_alive_groups_after=2)]),
        ]
    )

    # --- assert -----------------------
    assert [(c.executed_by, c.elapsed.t_elapsed_sec, c.n_alive_groups_after) for c in timeline.group_changes] == [
        (1, 3.0, 2),
        (0, 5.0, 1),
    ]
    assert [c.elapsed.n_iterations for c in timeline.group_changes] == [10, 50]  # counts stay the executor's own


def test_the_group_count_orders_changes_whose_clock_readings_disagree():
    """Two changes moments apart keep their true order even when the later executor's reading is earlier."""
    # --- arrange ----------------------
    timeline = SharedSolveTimeline.from_worker_results(
        [
            _result(0, t_start=100.0, changes=[_change(0, t=4.0, n_alive_groups_after=2)]),
            _result(1, t_start=100.0, changes=[_change(1, t=3.9, n_alive_groups_after=1)]),
        ]
    )

    # --- assert -----------------------
    assert [c.n_alive_groups_after for c in timeline.group_changes] == [2, 1]


def test_workers_without_changes_have_an_empty_group_history():
    """A solve whose workers executed no dissolution has no group changes."""
    # --- arrange / act ----------------
    timeline = SharedSolveTimeline.from_worker_results([_result(0, t_start=100.0), _result(1, t_start=101.0)])

    # --- assert -----------------------
    assert timeline.group_changes == []
