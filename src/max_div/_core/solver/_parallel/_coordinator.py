"""A worker calls its coordinator at each batch boundary, whether or not it has anything to say.

The call happens on every batch even when it does nothing, so the one place a worker could reach
the rest of the portfolio from is a path that runs rather than one that merely exists.
"""

from abc import ABC, abstractmethod

from max_div._core.solver._solver_state import SolverState


class WorkerCoordinator(ABC):
    """A worker uses its coordinator to reach the other workers solving the same problem."""

    @abstractmethod
    def at_batch_boundary(self, state: SolverState) -> None:
        """React to a worker finishing a batch, with the state it holds at that moment.

        Called on every batch of every optimization step, so keep an implementation cheap.
        """


class IndependentCoordinator(WorkerCoordinator):
    """Workers that share nothing hold this coordinator, and every call on it does nothing."""

    def at_batch_boundary(self, state: SolverState) -> None:
        """Do nothing: an independent worker has nobody to tell and nothing to ask."""
