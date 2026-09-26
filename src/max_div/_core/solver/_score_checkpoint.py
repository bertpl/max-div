from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._duration import Elapsed
    from ._parallel import WorkerCoordinator
    from ._score import Score
    from ._solver_state import SolverState
    from ._step_identity import SolverStepIdentity


# =================================================================================================
#  ScoreCheckpoint
# =================================================================================================
@dataclass(frozen=True)
class ScoreCheckpoint:
    """A checkpoint is a point-in-time summary of the score a solver / worker obtained so far.

    It captures score, the active solver step, elapsed time/iterations and worker/group index in case of
    a multi-worker setup.

    - `elapsed` is measured from the start of whatever produced the checkpoint:
      - a single step counts from its own start;
      - a whole solve counts from the solve's start (see `SolveTimeline`);
      - a parallel solve counts from its earliest worker's start (see `SharedSolveTimeline`).
    - `worker_index` and `group_index` name the parallel worker that recorded the checkpoint and the
      worker group it belonged to at that moment; both are `None` for a single (non-parallel) solve.
    - `i_selected` is the selection held at that moment as ascending indices, only when the solve was built
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
        state: SolverState,
        coordinator: WorkerCoordinator | None,
        includes_selection: bool = False,
    ) -> ScoreCheckpoint:
        """Return a checkpoint of the state's score, tagged with the worker and group that `coordinator` belongs to.

        The checkpoint stays untagged when `coordinator` is `None`. With `includes_selection` it also
        holds a copy of the state's selection, since the solver keeps mutating the state's own array.
        """
        i_selected = state.selected_index_array.copy() if includes_selection else None
        if coordinator is None:
            return cls(step_identity=step_identity, elapsed=elapsed, score=state.score, i_selected=i_selected)
        else:
            return cls(
                step_identity=step_identity,
                elapsed=elapsed,
                score=state.score,
                worker_index=coordinator.worker_index,
                group_index=coordinator.group_index,
                i_selected=i_selected,
            )

    def shifted_by(self, offset: Elapsed) -> ScoreCheckpoint:
        """Return this checkpoint with `offset` added to its elapsed time and iterations."""
        return replace(self, elapsed=offset + self.elapsed)
