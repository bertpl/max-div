from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._duration import Elapsed
    from ._parallel import WorkerCoordinator
    from ._score import Score
    from ._step_identity import SolverStepIdentity


# =================================================================================================
#  ScoreCheckpoint
# =================================================================================================
@dataclass(frozen=True)
class ScoreCheckpoint:
    """A checkpoint is a point-in-time summary of the score a solver / worker obtained so far.

    It captures score, the active solver step, elapsed time/iterations and worker/group index in case of
    a multi-worker setup.

    - `elapsed` is measured from the start of whatever produced the checkpoint: a single step counts
      from its own start, a whole solve from its first step.
    - `worker_index` and `group_index` name the parallel worker that recorded the checkpoint and the
      worker group it belonged to at that moment; both are `None` for a single (non-parallel) solve.
    - `i_selected` is the selection held at that moment, ascending, only when the solve was built
      with `with_intermediate_selections()`; `None` otherwise, as it costs k integers per checkpoint.
    """

    step_identity: SolverStepIdentity
    elapsed: Elapsed
    score: Score
    worker_index: int | None = None
    group_index: int | None = None
    i_selected: NDArray[np.int32] | None = None

    @classmethod
    def new(
        cls,
        step_identity: SolverStepIdentity,
        elapsed: Elapsed,
        score: Score,
        coordinator: WorkerCoordinator | None,
        i_selected: NDArray[np.int32] | None = None,
    ) -> ScoreCheckpoint:
        """Return a checkpoint tagged with the worker and group that `coordinator` belongs to.

        The checkpoint stays untagged when `coordinator` is `None`. `i_selected` is stored as given,
        so pass a copy of a selection that the solver keeps mutating.
        """
        if coordinator is None:
            return cls(step_identity=step_identity, elapsed=elapsed, score=score, i_selected=i_selected)
        else:
            return cls(
                step_identity=step_identity,
                elapsed=elapsed,
                score=score,
                worker_index=coordinator.worker_index,
                group_index=coordinator.group_index,
                i_selected=i_selected,
            )
