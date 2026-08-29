"""A worker calls its coordinator at every batch boundary; this module is the topology reference.

How the pieces connect, in every setup:

- **One coordinator per worker**, handed to the worker process at spawn.  Workers never talk to
  each other directly; a coordinator is their only sideways channel.
- **Independent workers** hold the no-op `IndependentCoordinator`.
- **A fixed worker group's members** each hold a `CooperativeCoordinator` bound to the group's one
  `GroupExchangeSlot`; groups never share slots, so no information crosses group boundaries.
- **A dynamic solve's workers** each hold a `DynamicGroupCoordinator` bound to one shared
  `DynamicGroupState`, which regroups the workers mid-solve (see `_dynamic_groups`).
- **Progress reporting is a separate channel entirely** — the one-way queue from workers to the
  parent (see `_progress_channel`).  Coordinators carry search information sideways between a
  group's workers and never progress; the queue carries progress up to the parent and never
  search information.
"""

from abc import ABC, abstractmethod

from max_div._core.solver._solver_state import SolverState

from ._exchange_slot import GroupExchangeSlot


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
                the grouping schedule run inside the workers.
        """


class IndependentCoordinator(WorkerCoordinator):
    """Workers that share nothing hold this coordinator."""

    def at_batch_boundary(self, state: SolverState, progress_fraction: float) -> None:
        """Do nothing: an independent worker has nobody to tell and nothing to ask."""


class CooperativeCoordinator(WorkerCoordinator):
    """The members of one fixed worker group hold this coordinator, all bound to the group's shared slot."""

    def __init__(self, slot: GroupExchangeSlot) -> None:
        """Bind the coordinator to its worker group's exchange slot."""
        self._slot = slot

    def at_batch_boundary(self, state: SolverState, progress_fraction: float) -> None:
        """Publish this worker's selection if it is the group's best, else adopt a strictly better one.

        The slot visit is a single lock acquisition; the adoption — the expensive part — runs
        after the lock is released, on a copy the slot handed back.
        """
        incoming = self._slot.exchange(state.score.as_tuple(), state.selected_index_array)
        if incoming is not None:
            state.adopt_selection(incoming)
