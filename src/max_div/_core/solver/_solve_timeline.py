"""A solve timeline places every step of one solve on the solve-wide time axis, whose zero is the solve's start.

Each step measures its elapsed time from its own start. The timeline reads the monotonic clock when
the solve starts and again when each step starts, so the axis measures real time: the axis also
counts the solver's own work before and between steps, which no step timer covers.

A parallel solve runs one solve timeline in each worker; `SharedSolveTimeline` then places the
workers on one axis.
"""

import time

from ._duration import Elapsed
from ._score_checkpoint import ScoreCheckpoint
from ._solver_step import SolverStepResult


class SolveTimeline:
    """A solve timeline records each step's duration, and the solve's checkpoints on the solve-wide axis.

    A solve calls `record_step_start` as each step starts and `record_step_result` with its result,
    in step order.
    """

    def __init__(self) -> None:
        """Start the solve-wide axis now."""
        self._t_start = time.monotonic()
        self._step_durations: list[Elapsed] = []
        self._checkpoints: list[ScoreCheckpoint] = []
        self._elapsed_before_step: Elapsed | None = None  # set by record_step_start, cleared by record_step_result

    @property
    def step_durations(self) -> list[Elapsed]:
        """Return each finished step's own duration, in step order."""
        return list(self._step_durations)

    @property
    def checkpoints(self) -> list[ScoreCheckpoint]:
        """Return every finished step's checkpoints, on the solve-wide axis and in step order."""
        return list(self._checkpoints)

    def record_step_start(self) -> Elapsed:
        """Mark a step as starting now, and return where it starts on the solve-wide axis.

        The time is measured since the solve started, not summed from the earlier steps' durations,
        so the time also counts the solver's own work before and between steps; the iteration count
        is the earlier steps' total.
        """
        self._elapsed_before_step = Elapsed(
            t_elapsed_sec=time.monotonic() - self._t_start,
            n_iterations=sum(duration.n_iterations for duration in self._step_durations),
        )
        return self._elapsed_before_step

    def record_step_result(self, step_result: SolverStepResult) -> None:
        """Record the step that started last: its duration, and its checkpoints shifted onto the solve-wide axis.

        Raises:
            RuntimeError: If no `record_step_start` call precedes this one, so the step's start is unknown.
        """
        if self._elapsed_before_step is None:
            raise RuntimeError("record_step_result() needs a matching record_step_start() first.")
        self._step_durations.append(step_result.elapsed)
        self._checkpoints.extend(
            checkpoint.shifted_by(self._elapsed_before_step) for checkpoint in step_result.score_checkpoints
        )
        self._elapsed_before_step = None
