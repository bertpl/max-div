"""A worker calls its coordinator at every batch boundary; this module is the topology reference.

How the pieces connect, in every setup:

- **One coordinator per worker**, handed to the worker process at spawn.  Workers never talk to
  each other directly; a coordinator is their only sideways channel.
- **Independent workers** hold the no-op `IndependentCoordinator`.
- **A worker group's members** each hold a `CooperativeCoordinator` bound to the group's one
  `GroupIncumbentSlot`; groups never share slots, so no information crosses group boundaries.
- **Progress reporting is a separate channel entirely** — the one-way queue from workers to the
  parent (see `_progress_channel`).  Coordinators carry search information sideways between a
  group's workers and never progress; the queue carries progress up to the parent and never
  search information.
"""

from abc import ABC, abstractmethod

from max_div._core.solver._solver_state import SolverState

from ._incumbent_slot import GroupIncumbentSlot


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
    """The members of one worker group hold this coordinator, all bound to the group's shared slot."""

    def __init__(self, slot: GroupIncumbentSlot) -> None:
        """Bind the coordinator to its worker group's incumbent slot."""
        self._slot = slot

    def at_batch_boundary(self, state: SolverState) -> None:
        """Publish this worker's selection if it is the group's best, else adopt a strictly better one.

        The slot visit is a single lock acquisition; the adoption — the expensive part — runs
        after the lock is released, on a copy the slot handed back.
        """
        incoming = self._slot.exchange(state.score.as_tuple(), state.selected_index_array)
        if incoming is not None:
            state.adopt_selection(incoming)
