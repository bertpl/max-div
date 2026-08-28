"""Adaptive worker groups: every worker starts in its own group, and groups consolidate toward one.

The group count decreases linearly from `n_workers` to 1 over the progress fraction of the solve's
time budget.  Each decrease dissolves the group whose incumbent slot holds the worst score —
the group whose best score is lowest so far — and reassigns its workers to the strongest groups that are
short a member, so they reinforce searches that can still win.

The module adds three pieces on top of the fixed-group code in `_coordinator` / `_incumbent_slot`:

- **One slot per worker**, allocated up front; a group is the set of workers currently assigned to
  one slot, and dissolving a group simply stops assigning workers to its slot.
- **A shared assignment table** maps each worker to its slot.  Workers read their entry at every
  batch boundary and exchange with the slot it names; the orchestrator is the only writer.
- **The orchestrator runs in the parent**, on a thread beside the result-draining loop.  It ranks
  groups on their slots at each dissolution, which the slots' publish-if-better semantics make
  safe: a slot holds its group's best score so far, so dissolving the worst-slot group can never
  discard the overall best selection.

A worker learns of its reassignment at its next batch boundary, so membership changes are eventual
rather than synchronized — a worker mid-batch keeps exchanging with its old slot for at most one
more batch.
"""

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from max_div._core.solver._duration import E2eBudget
from max_div._core.solver._solver_state import SolverState

from ._coordinator import WorkerCoordinator
from ._incumbent_slot import GroupIncumbentSlot

if TYPE_CHECKING:
    import ctypes
    from multiprocessing.context import BaseContext

# The orchestrator re-checks the schedule this often.  It only bounds how late a dissolution can
# fire after its scheduled progress fraction; small enough to be negligible against even
# sub-second budgets, and its cost is a handful of shared-memory reads.
_TICK_SECONDS = 0.05


def adaptive_group_count(n_workers: int, progress_fraction: float) -> int:
    """Return the scheduled group count at the given progress fraction.

    The count decreases linearly from `n_workers` at fraction 0 and reaches 1 at fraction
    `(n_workers - 1) / n_workers`, so every group count holds for an equal share of the budget.
    """
    fraction = min(max(progress_fraction, 0.0), 1.0)
    return max(1, math.ceil(n_workers * (1.0 - fraction)))


@dataclass(frozen=True)
class DissolutionEvent:
    """A record of one group dissolution, kept for inspecting the mechanism after a solve.

    Args:
        t_sec: seconds since the budget's start when the dissolution fired.
        progress_fraction: budget fraction spent at that moment.
        dissolved_group: index of the dissolved group's slot.
        slot_scores: every then-alive group's slot score, None for a never-written slot.
        reassignments: target group per freed worker.
    """

    t_sec: float
    progress_fraction: float
    dissolved_group: int
    slot_scores: dict[int, tuple[float, ...] | None]
    reassignments: dict[int, int]


class AdaptiveGroupCoordinator(WorkerCoordinator):
    """A worker of an adaptive solve exchanges with whichever slot the assignment table names."""

    def __init__(
        self, slots: list[GroupIncumbentSlot], assignment: "ctypes.Array[ctypes.c_int]", worker_index: int
    ) -> None:
        """Bind the coordinator to the shared slots and the worker's assignment-table entry."""
        self._slots = slots
        self._assignment = assignment
        self._worker_index = worker_index

    def at_batch_boundary(self, state: SolverState) -> None:
        """Exchange with the currently assigned slot, adopting a strictly better incumbent.

        The assignment read is one shared-memory element; re-reading it every boundary keeps the
        coordinator free of any synchronization with the orchestrator's writes.
        """
        slot = self._slots[self._assignment[self._worker_index]]
        incoming = slot.exchange(state.score.as_tuple(), state.selected_index_array)
        if incoming is not None:
            state.adopt_selection(incoming)


class AdaptiveGroupOrchestrator:
    """The orchestrator owns the schedule: it dissolves groups as the budget's progress advances.

    Created in the parent before workers spawn; `run` executes on a parent thread while the
    workers solve, and `events` afterwards holds every dissolution for inspection.
    """

    def __init__(self, context: "BaseContext", n_workers: int, k: int, score_length: int) -> None:
        """Allocate one slot per worker and the identity assignment, all in shared memory.

        Args:
            n_workers: the worker count, fixing the slot and assignment-table sizes.
            k: maximum selection size the slots hold.
            score_length: number of components in the workers' score tuples.
        """
        self._n_workers = n_workers
        self._slots = [GroupIncumbentSlot(context, k=k, score_length=score_length) for _ in range(n_workers)]
        self._assignment = context.Array("i", list(range(n_workers)), lock=False)
        # `_members` is parent-local; the orchestrator thread is its only mutator
        self._members: dict[int, list[int]] = {index: [index] for index in range(n_workers)}
        self.events: list[DissolutionEvent] = []

    def coordinator_for(self, worker_index: int) -> AdaptiveGroupCoordinator:
        """Return the given worker's coordinator, bound to the shared slots and assignment table."""
        return AdaptiveGroupCoordinator(self._slots, self._assignment, worker_index)

    def run(self, budget: E2eBudget, stop: threading.Event) -> None:
        """Dissolve groups per the schedule until the budget is spent or `stop` is set.

        Runs on a parent thread beside the executor's drain loop; exits on its own once the
        schedule reaches one group, since no further dissolution can fire.
        """
        while not stop.is_set():
            spent_sec = budget.budget_sec - budget.remaining_sec()
            fraction = spent_sec / budget.budget_sec
            target = adaptive_group_count(self._n_workers, fraction)
            while len(self._members) > target:
                self._dissolve_worst(spent_sec, fraction)
            if len(self._members) == 1:
                return
            stop.wait(_TICK_SECONDS)

    def _dissolve_worst(self, t_sec: float, fraction: float) -> None:
        """Dissolve the worst-scoring group and reassign its workers to the strongest short groups."""
        scores = {group: self._slots[group].peek_score() for group in self._members}
        # a never-written slot ranks below any written one (the empty tuple sorts below any real
        # score tuple); ties go to the lowest group index
        worst = min(self._members, key=lambda group: (scores[group] or (), group))
        freed = self._members.pop(worst)
        reassignments: dict[int, int] = {}
        for worker in freed:
            target = self._reassignment_target(scores)
            self._members[target].append(worker)
            self._assignment[worker] = target
            reassignments[worker] = target
        self.events.append(
            DissolutionEvent(
                t_sec=t_sec,
                progress_fraction=fraction,
                dissolved_group=worst,
                slot_scores=scores,
                reassignments=reassignments,
            )
        )

    def _reassignment_target(self, scores: dict[int, tuple[float, ...] | None]) -> int:
        """Return the group a freed worker joins: the best-scoring one still short of the largest.

        When every surviving group has the same size, no group is short and the best-scoring one
        overall takes the worker.
        """
        largest = max(len(members) for members in self._members.values())
        short = [group for group, members in self._members.items() if len(members) < largest]
        pool = short if short else list(self._members)
        return max(pool, key=lambda group: (scores.get(group) or (), -group))
