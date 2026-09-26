"""A solve timeline places every step of one solve on the solve-wide time axis, whose zero is the solve's start.

Each step counts elapsed from its own start. The timeline reads the monotonic clock when the solve
starts and again when each step starts, so the axis measures real time: it also counts the solver's
own work before and between steps, which no step timer covers. `SharedSolveTimeline` places the
workers of a parallel solve on one axis in the same way.
"""

import time

from ._duration import Elapsed
from ._score_checkpoint import ScoreCheckpoint
from ._solver_step import SolverStepResult


class SolveTimeline:
    """A solve timeline records one solve's step durations, and its checkpoints on the solve-wide axis.

    A solve calls `start_step` as each step starts and `finish_step` with its result, in step order;
    the solver state initialization is step 0.
    """

    def __init__(self) -> None:
        """Start the solve-wide axis now."""
        self._t_start = time.monotonic()
        self._step_durations: list[Elapsed] = []
        self._checkpoints: list[ScoreCheckpoint] = []
        self._elapsed_before_step = Elapsed(t_elapsed_sec=0.0, n_iterations=0)

    @property
    def step_durations(self) -> list[Elapsed]:
        """Return each finished step's own duration, in step order."""
        return list(self._step_durations)

    @property
    def checkpoints(self) -> list[ScoreCheckpoint]:
        """Return every finished step's checkpoints, on the solve-wide axis and in step order."""
        return list(self._checkpoints)

    def start_step(self) -> Elapsed:
        """Mark a step as starting now, and return where it starts on the solve-wide axis.

        The time is measured since the solve started, not summed from the earlier steps' durations,
        so it also counts the solver's own work before and between steps; the iteration count is the
        earlier steps' total.
        """
        self._elapsed_before_step = Elapsed(
            t_elapsed_sec=time.monotonic() - self._t_start,
            n_iterations=sum(duration.n_iterations for duration in self._step_durations),
        )
        return self._elapsed_before_step

    def finish_step(self, step_result: SolverStepResult) -> None:
        """Record the step started last: its duration, and its checkpoints shifted onto the solve-wide axis."""
        self._step_durations.append(step_result.elapsed)
        self._checkpoints.extend(
            checkpoint.shifted_by(self._elapsed_before_step) for checkpoint in step_result.score_checkpoints
        )
