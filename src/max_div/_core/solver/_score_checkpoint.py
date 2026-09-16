from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._duration import Elapsed
    from ._parallel import WorkerCoordinator
    from ._score import Score
    from ._step_identity import SolverStepIdentity


# =================================================================================================
#  ScoreCheckpoint
# =================================================================================================
@dataclass(frozen=True)
class ScoreCheckpoint:
    """A checkpoint records the score a solve held at one moment, in which step, and which parallel worker held it.

    `elapsed` is measured from the start of whatever produced the checkpoint: a single step counts
    from its own start, a whole solve from its first step. `worker_index` and `group_index` name
    the parallel worker that recorded the checkpoint and the worker group it belonged to at that
    moment; both are `None` for a single (non-parallel) solve.
    """

    step_identity: SolverStepIdentity
    elapsed: Elapsed
    score: Score
    worker_index: int | None = None
    group_index: int | None = None

    @classmethod
    def new(
        cls,
        step_identity: SolverStepIdentity,
        elapsed: Elapsed,
        score: Score,
        coordinator: WorkerCoordinator | None,
    ) -> ScoreCheckpoint:
        """Return a checkpoint tagged with the worker and group `coordinator` belongs to; untagged when it is `None`."""
        if coordinator is None:
            return cls(step_identity=step_identity, elapsed=elapsed, score=score)
        else:
            return cls(
                step_identity=step_identity,
                elapsed=elapsed,
                score=score,
                worker_index=coordinator.worker_index,
                group_index=coordinator.group_index,
            )
