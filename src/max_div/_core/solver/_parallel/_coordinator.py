"""A worker calls its coordinator at every batch boundary; this module holds the contract.

Every parallel worker holds a `WorkerGroupCoordinator` (see `_worker_groups`) bound to the
solve's one shared `WorkerGroupState` — the coordinator is a worker's only sideways channel.
Progress reporting is a separate channel entirely: the one-way queue from workers to the parent
(see `_progress_channel`). Coordinators carry search information sideways between workers and
never progress; the queue carries progress up to the parent and never search information.
"""

from abc import ABC, abstractmethod

from max_div._core.solver._solver_state import SolverState


class WorkerCoordinator(ABC):
    """A worker uses its coordinator to reach the other workers solving the same problem."""

    @abstractmethod
    def at_batch_boundary(self, state: SolverState, progress_fraction: float) -> None:
        """React to a worker finishing a batch, with the state the worker holds at that moment.

        Called on every batch of every optimization step, so keep an implementation cheap.

        Args:
            state: the worker's mutable solver state at this boundary.
            progress_fraction: the worker's own progress through its optimization step, 0 to 1 —
                meaningful under time and iteration budgets alike, which lets
                `WorkerGroupCoordinator` run its regrouping schedule inside the
                workers (see `_worker_groups`).
        """
