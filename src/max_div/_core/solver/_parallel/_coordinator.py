"""The handle a worker holds on the rest of the portfolio, called at every batch boundary.

Workers reach this seam whether or not they use it.  Independent workers never exchange anything,
so the coordinator they hold does nothing — but the call still happens on every batch, which keeps
the seam exercised rather than merely present, and leaves one place for a mode that does share to
publish its selection or consult someone else's.

The batch boundary is where the optimization step already pauses to record a score checkpoint, so
sharing costs a worker no additional interruption.
"""

from abc import ABC, abstractmethod

from max_div._core.solver._solver_state import SolverState


class WorkerCoordinator(ABC):
    """What a worker may tell the rest of the portfolio, and ask of it, between batches."""

    @abstractmethod
    def at_batch_boundary(self, state: SolverState) -> None:
        """React to a worker reaching the end of a batch, with its current state in hand.

        Called on every batch of every optimization step, so an implementation that does real work
        is responsible for its own cost.
        """


class IndependentCoordinator(WorkerCoordinator):
    """The coordinator of workers that share nothing: every call is a no-op.

    Independent solving is the whole of what this class expresses — each worker runs its own seeded
    search start to finish, and the portfolio compares only the results.
    """

    def at_batch_boundary(self, state: SolverState) -> None:
        """Do nothing: an independent worker has nobody to tell and nothing to ask."""
