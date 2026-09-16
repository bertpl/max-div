"""The parent rebuilds, from every worker's own checkpoints, the best score any worker held at each moment.

A worker's checkpoints trace that worker's own solve, so the winning worker's trace steps up whenever
it adopts its group's best: progress made by another worker shows up as a jump at the adoption.
The trace built here follows the best score across all workers instead, and each of its checkpoints
names the worker that held the score, so a jump is attributed to the worker that made it.

Every worker counts elapsed time from its own start, and the workers start at slightly different
moments. Each worker reports its start as a `time.monotonic()` reading, which is one systemwide clock
on the platforms max-div runs on, so the parent can place all checkpoints on one axis whose zero is the
earliest worker start. The per-worker offsets are kept: a worker that started first genuinely searched
alone until the others existed.
"""

from dataclasses import replace

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score_checkpoint import ScoreCheckpoint

from ._result import WorkerResult, earliest_start_time


def best_known_trajectory(results: list[WorkerResult]) -> list[ScoreCheckpoint]:
    """Return the checkpoints that trace the best score any worker held at each moment, in time order.

    The result keeps every checkpoint that improves on the best score so far, and ends with the
    last checkpoint that holds the final best score, so its last entry spans the whole solve and
    the solution's `duration` reads the full span. Each checkpoint's `elapsed` counts from the
    earliest worker start, and its iteration count stays the holding worker's own.

    Args:
        results: what each worker reported; every result's `t_start` places its checkpoints. Non-empty,
            and at least one checkpoint across all results.
    """
    merged_checkpoints = _merge_checkpoints(results)
    trajectory: list[ScoreCheckpoint] = []
    for checkpoint in merged_checkpoints:
        if (not trajectory) or (trajectory[-1].score < checkpoint.score):
            trajectory.append(checkpoint)
    # close the trace at the end of the solve: the latest checkpoint that still holds the best score
    # (max returns the first of equal times, which is the lowest worker index in the merged order)
    best_score = trajectory[-1].score
    closing = max(
        (checkpoint for checkpoint in merged_checkpoints if checkpoint.score == best_score),
        key=lambda checkpoint: checkpoint.elapsed.t_elapsed_sec,
    )
    if closing is not trajectory[-1]:
        trajectory.append(closing)
    return trajectory


def _merge_checkpoints(results: list[WorkerResult]) -> list[ScoreCheckpoint]:
    """Return every worker's checkpoints with `elapsed` counted from the earliest worker start, in time order.

    Ties on time resolve by worker index, so the same results give the same order whatever the list order.
    """
    t_first_start = earliest_start_time(results)
    merged_checkpoints = [
        replace(
            checkpoint,
            elapsed=Elapsed(
                t_elapsed_sec=(result.t_start - t_first_start) + checkpoint.elapsed.t_elapsed_sec,
                n_iterations=checkpoint.elapsed.n_iterations,
            ),
        )
        for result in results
        for checkpoint in result.solution.score_checkpoints
    ]
    return sorted(
        merged_checkpoints, key=lambda checkpoint: (checkpoint.elapsed.t_elapsed_sec, checkpoint.worker_index)
    )
