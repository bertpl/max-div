"""The parent maps every worker's events onto one time axis: the shared timeline of a parallel solve.

Each worker counts elapsed time from its own start, and the workers start at slightly different
moments. Each worker reports its start as a `time.monotonic()` reading, one systemwide clock on the
platforms max-div runs on, so the parent can place every worker's events on one axis whose zero is
the earliest worker start. The per-worker offsets are kept: a worker that started first genuinely
searched alone until the others started.

`SharedSolveTimeline.from_worker_results` shifts every worker's events onto that axis once, so the
consumers downstream read three things off it without shifting again: the worker start offsets, the
ordered group history, and the best-known checkpoint trace.
"""

from dataclasses import dataclass, replace

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score_checkpoint import ScoreCheckpoint

from ._result import WorkerResult
from ._worker_group_change import WorkerGroupChange


@dataclass(frozen=True)
class SharedSolveTimeline:
    """A shared solve timeline holds every worker's events on one time axis whose zero is the earliest worker start.

    - `checkpoints`: all workers' checkpoints, their `elapsed` fields rebased to the shared axis, in no
      significant order; `best_known_checkpoints` derives the best-known trace from them.
    - `group_changes`: all workers' dissolutions, rebased and already in the order they happened, so the
      field is the group history itself.
    - `start_offsets`: maps a worker index to its start on the axis, zero for the earliest worker.
    """

    start_offsets: dict[int, float]
    checkpoints: list[ScoreCheckpoint]
    group_changes: list[WorkerGroupChange]

    @classmethod
    def from_worker_results(cls, results: list[WorkerResult]) -> "SharedSolveTimeline":
        """Shift every worker's checkpoints and group changes onto the shared axis.

        Args:
            results: what each worker reported; every result's `t_start` places its events. Non-empty.
        """
        t_first_start = WorkerResult.earliest_start_time(results)
        start_offsets = {result.worker_index: result.t_start - t_first_start for result in results}
        checkpoints = [
            cls._shifted_onto_axis(checkpoint, start_offsets[result.worker_index])
            for result in results
            for checkpoint in result.solution.score_checkpoints
        ]
        group_changes = [
            cls._shifted_onto_axis(change, start_offsets[result.worker_index])
            for result in results
            for change in result.worker_group_changes
        ]
        # every dissolution lowers the alive-group count by one, and each worker sets that count as it
        # makes the change, so sorting on the count orders the changes as they happened across workers.
        # A per-worker timestamp could not, because a worker reads it before the change takes effect.
        group_changes.sort(key=lambda change: -change.n_alive_groups_after)
        return cls(start_offsets=start_offsets, checkpoints=checkpoints, group_changes=group_changes)

    def best_known_checkpoints(self) -> list[ScoreCheckpoint]:
        """Return the checkpoints that trace the best score any worker held at each moment, in time order.

        The trace keeps every checkpoint that improves on the best score so far, and ends with the last
        checkpoint that holds the final best score, so its last entry spans the whole solve and the
        solution's `duration` reads the full span. A checkpoint's iteration count stays the holding
        worker's own. Requires at least one checkpoint.
        """
        ordered = sorted(self.checkpoints, key=lambda c: (c.elapsed.t_elapsed_sec, c.worker_index))
        trace: list[ScoreCheckpoint] = []
        for checkpoint in ordered:
            if (not trace) or (trace[-1].score < checkpoint.score):
                trace.append(checkpoint)
        # close the trace at the end of the solve: the latest checkpoint that still holds the best
        # score (max returns the first of equal times, which is the lowest worker index in `ordered`)
        best_score = trace[-1].score
        closing = max(
            (checkpoint for checkpoint in ordered if checkpoint.score == best_score),
            key=lambda checkpoint: checkpoint.elapsed.t_elapsed_sec,
        )
        if closing is not trace[-1]:
            trace.append(closing)
        return trace

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _shifted_onto_axis[T: (ScoreCheckpoint, WorkerGroupChange)](event: T, offset_sec: float) -> T:
        """Move `event`'s `elapsed` field onto the shared axis by `offset_sec`, keeping its iteration count."""
        return replace(
            event,
            elapsed=Elapsed(
                t_elapsed_sec=offset_sec + event.elapsed.t_elapsed_sec, n_iterations=event.elapsed.n_iterations
            ),
        )
