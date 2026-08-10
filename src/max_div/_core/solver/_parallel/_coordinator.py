"""A worker calls its coordinator at every batch boundary.

Independent workers do nothing there; the call exists so cooperative parallelism can reuse this
path rather than add one.
"""

from abc import ABC, abstractmethod

from max_div._core.solver._solver_state import SolverState


class WorkerCoordinator(ABC):
    """A worker uses its coordinator to reach the other workers solving the same problem."""

    @abstractmethod
    def at_batch_boundary(self, state: SolverState) -> None:
        """React to a worker finishing a batch, with the state the worker holds at that moment.

        Called on every batch of every optimization step, so keep an implementation cheap.
        """


class IndependentCoordinator(WorkerCoordinator):
    """Workers that share nothing hold this coordinator."""

    def at_batch_boundary(self, state: SolverState) -> None:
        """Do nothing: an independent worker has nobody to tell and nothing to ask."""
