"""A worker calls its coordinator at every batch boundary; this module holds the contract.

Two channels leave a worker, each carrying one kind of information:

- **the coordinator** carries only search information, sideways between workers — every parallel
  worker holds a `WorkerGroupCoordinator` (see `_worker_groups`) bound to the solve's one shared
  `WorkerGroupState`;
- **the progress queue** (see `_progress_channel`) carries only progress, one-way up to the
  parent.
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
