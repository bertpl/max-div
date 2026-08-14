"""A worker calls its coordinator at every batch boundary.

Independent workers do nothing there; cooperative workers exchange their selection with the
island's shared incumbent slot.
"""

from abc import ABC, abstractmethod

from max_div._core.solver._solver_state import SolverState

from ._incumbent_slot import IslandIncumbentSlot


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


class CooperativeCoordinator(WorkerCoordinator):
    """The members of one island hold this coordinator, all bound to the island's shared slot."""

    def __init__(self, slot: IslandIncumbentSlot) -> None:
        """Bind the coordinator to its island's incumbent slot."""
        self._slot = slot

    def at_batch_boundary(self, state: SolverState) -> None:
        """Publish this worker's selection if it is the island's best, else adopt a strictly better one.

        The slot visit is a single lock acquisition; the adoption — the expensive part — runs
        after the lock is released, on a copy the slot handed back.
        """
        incoming = self._slot.exchange(state.score.as_tuple(), state.selected_index_array)
        if incoming is not None:
            state.adopt_selection(incoming)
