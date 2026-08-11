from collections.abc import Callable
from dataclasses import replace

import numpy as np

from max_div._core.solver._duration import Elapsed, Progress
from max_div._core.solver._parallel._progress_view import ParallelProgressView
from max_div._core.solver._parallel._result import WorkerResult
from max_div._core.solver._progress_reporting import ProgressReporter, ProgressSnapshot
from max_div._core.solver._score import Score
from max_div._core.solver._solution import MaxDivSolution


class _RecordingReporter(ProgressReporter):
    """A recording reporter stores every rendered snapshot, so tests can inspect the composites."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, ProgressSnapshot | str]] = []

    def show_step_started(self, step_name: str) -> None:
        self.events.append(("started", step_name))

    def show_update(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        self.events.append(("update", snapshot))

    def show_step_finished(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        self.events.append(("finished", snapshot))

    def show_milestone(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        self.events.append(("milestone", snapshot))

    def updates(self) -> list[ProgressSnapshot]:
        return [snapshot for kind, snapshot in self.events if kind == "update"]  # ty: ignore[invalid-return-type]


def _snapshot(worker_index: int, fraction: float, diversity: float, iter_count: int = 10) -> ProgressSnapshot:
    """Return a materialized snapshot, as a worker would forward it."""
    return ProgressSnapshot(
        step_name="",
        progress=Progress(
            tqdm_n_total=100,
            fraction=fraction,
            iter_count=iter_count,
            est_n_iters_remaining=0,
            est_iters_per_second=0.0,
        ),
        t_elapsed_solver=1.0,
        t_elapsed_step=1.0,
        score=Score(size=1.0, constraints=1.0, diversity=diversity, div_tie_breakers=()),
        n_selected=5,
        k=5,
        m=0,
        selection=None,
        ignore_infeasible_diversity=False,
        selection_hash="abcd",
        worker_index=worker_index,
    )


def _result(worker_index: int) -> WorkerResult:
    """Return a minimal result marking `worker_index` as finished."""
    solution = MaxDivSolution(
        i_selected=np.arange(5, dtype=np.int32),
        score_checkpoints=[("step", Elapsed(t_elapsed_sec=1.0, n_iterations=10), _snapshot(0, 1.0, 0.0).score)],
        step_durations={"step": Elapsed(t_elapsed_sec=1.0, n_iterations=10)},
        n_constraints=0,
        n_constraints_satisfied=0,
        distance_storage="",
    )
    return WorkerResult(worker_index=worker_index, seed=0, solution=solution)


def test_progress_follows_the_slowest_live_worker():
    """The composite fraction is the minimum across workers that are still running."""
    # --- arrange -----------------------------------------
    reporter = _RecordingReporter()
    view = ParallelProgressView(reporter, n_workers=2)
    view.start()

    # --- act ---------------------------------------------
    view.on_snapshot(_snapshot(worker_index=0, fraction=0.8, diversity=0.5))
    view.on_snapshot(_snapshot(worker_index=1, fraction=0.3, diversity=0.4))

    # --- assert ------------------------------------------
    last = reporter.updates()[-1]
    assert last.progress.fraction == 0.3
    assert last.n_active == 2


def test_result_half_is_the_best_so_far_and_never_regresses():
    """A finished best worker keeps the result columns; slower workers only take over by beating it."""
    # --- arrange -----------------------------------------
    reporter = _RecordingReporter()
    view = ParallelProgressView(reporter, n_workers=2)
    view.start()
    view.on_snapshot(_snapshot(worker_index=0, fraction=1.0, diversity=0.9))
    view.on_worker_finished(_result(0))  # best worker finishes first

    # --- act ---------------------------------------------
    view.on_snapshot(_snapshot(worker_index=1, fraction=0.5, diversity=0.4))  # worse, still running

    # --- assert ------------------------------------------
    last = reporter.updates()[-1]
    assert last.worker_index == 0  # the finished best still owns the result columns
    assert last.worker_finished is True
    assert last.score.diversity == 0.9
    assert last.progress.fraction == 0.5  # while progress follows the live worker
    assert last.n_active == 1


def test_better_late_result_takes_over():
    """A running worker that beats the finished best takes the result columns over."""
    # --- arrange -----------------------------------------
    reporter = _RecordingReporter()
    view = ParallelProgressView(reporter, n_workers=2)
    view.start()
    view.on_snapshot(_snapshot(worker_index=0, fraction=1.0, diversity=0.5))
    view.on_worker_finished(_result(0))

    # --- act ---------------------------------------------
    view.on_snapshot(_snapshot(worker_index=1, fraction=0.5, diversity=0.7))

    # --- assert ------------------------------------------
    last = reporter.updates()[-1]
    assert last.worker_index == 1
    assert last.score.diversity == 0.7
    assert last.worker_finished is False


def test_worker_finish_renders_a_milestone_with_its_own_state():
    """Finishing prints that worker's final state set off from the stream, marked finished."""
    # --- arrange -----------------------------------------
    reporter = _RecordingReporter()
    view = ParallelProgressView(reporter, n_workers=2)
    view.start()
    view.on_snapshot(_snapshot(worker_index=0, fraction=1.0, diversity=0.6))
    view.on_snapshot(_snapshot(worker_index=1, fraction=0.4, diversity=0.9))  # global best is worker 1

    # --- act ---------------------------------------------
    view.on_worker_finished(_result(0))

    # --- assert ------------------------------------------
    kind, milestone = reporter.events[-1]
    assert kind == "milestone"
    assert milestone.worker_index == 0  # the milestone shows the finisher itself, not the global best
    assert milestone.score.diversity == 0.6
    assert milestone.worker_finished is True


def test_dead_worker_is_dropped_from_the_progress_half():
    """A dead worker stops holding the minimum down, so the view keeps advancing."""
    # --- arrange -----------------------------------------
    reporter = _RecordingReporter()
    view = ParallelProgressView(reporter, n_workers=2)
    view.start()
    view.on_snapshot(_snapshot(worker_index=0, fraction=0.9, diversity=0.5))
    view.on_snapshot(_snapshot(worker_index=1, fraction=0.1, diversity=0.4))

    # --- act ---------------------------------------------
    view.on_worker_died(1)
    view.on_snapshot(_snapshot(worker_index=0, fraction=0.95, diversity=0.5))

    # --- assert ------------------------------------------
    last = reporter.updates()[-1]
    assert last.progress.fraction == 0.95
    assert last.n_active == 1


def test_finish_renders_a_full_progress_closing_row():
    """The closing row reports 100% progress and the best result."""
    # --- arrange -----------------------------------------
    reporter = _RecordingReporter()
    view = ParallelProgressView(reporter, n_workers=1)
    view.start()
    view.on_snapshot(_snapshot(worker_index=0, fraction=0.7, diversity=0.5))
    view.on_worker_finished(_result(0))

    # --- act ---------------------------------------------
    view.finish()

    # --- assert ------------------------------------------
    kind, closing = reporter.events[-1]
    assert kind == "finished"
    assert closing.progress.fraction == 1.0
    assert closing.n_active == 0
    assert closing.score.diversity == 0.5


def test_finish_with_no_snapshots_renders_nothing():
    """Every worker dying before its first snapshot leaves nothing to render, not a crash."""
    # --- arrange -----------------------------------------
    reporter = _RecordingReporter()
    view = ParallelProgressView(reporter, n_workers=1)
    view.start()
    view.on_worker_died(0)

    # --- act ---------------------------------------------
    view.finish()

    # --- assert ------------------------------------------
    assert [kind for kind, _ in reporter.events] == ["started"]


def test_unattributable_snapshot_is_ignored():
    """A snapshot without a worker index cannot be folded in and is dropped."""
    # --- arrange -----------------------------------------
    reporter = _RecordingReporter()
    view = ParallelProgressView(reporter, n_workers=1)
    view.start()

    # --- act ---------------------------------------------
    view.on_snapshot(replace(_snapshot(worker_index=0, fraction=0.5, diversity=0.5), worker_index=None))

    # --- assert ------------------------------------------
    assert reporter.updates() == []
