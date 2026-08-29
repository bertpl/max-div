"""Dynamic worker groups: every worker starts in its own group, and groups consolidate toward one.

The group count decreases linearly from `n_workers` to 1 over the progress fraction of each
worker's optimization step.  Each decrease dissolves the group whose exchange slot holds the
worst score — the group whose best score is lowest so far — and reassigns its workers to the
strongest groups that are short a member, so they reinforce searches that can still win.

The module adds two pieces on top of the fixed-group code in `_coordinator` / `_exchange_slot`;
the workers themselves run the schedule:

- **`DynamicGroupState` is the shared-memory record of the grouping**: one slot per worker, an
  assignment table mapping each worker to its slot, the alive group count, and the dissolution
  log.
- **`DynamicGroupCoordinator` runs the schedule from inside the workers**: at each batch
  boundary a worker computes the scheduled group count from its own progress fraction, and
  whichever worker first sees the alive count exceed the schedule executes the dissolution
  itself, under a single transition lock.

Ranking groups on their slots makes dissolution safe: a slot only ever accepts a strictly better
selection (`GroupExchangeSlot.exchange`), so it holds its group's best score so far, and
dissolving the worst-slot group can never discard the overall best selection.

A worker learns of its reassignment at its next batch boundary, so membership changes are eventual
rather than synchronized — a worker mid-batch keeps exchanging with its old slot for at most one
more batch.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from max_div._core.solver._solver_state import SolverState

from ._coordinator import WorkerCoordinator
from ._exchange_slot import GroupExchangeSlot

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    import numpy as np
    from numpy.typing import NDArray


def dynamic_group_count(n_workers: int, progress_fraction: float) -> int:
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
        progress_fraction: progress fraction of the worker that executed the dissolution.
        dissolved_group: index of the dissolved group's slot.
        slot_scores: every then-alive group's slot score, None for a never-written slot.
        reassignments: target group per freed worker.
    """

    progress_fraction: float
    dissolved_group: int
    slot_scores: dict[int, tuple[float, ...] | None]
    reassignments: dict[int, int]


class DynamicGroupState:
    """The shared group state records a dynamic solve's grouping and executes the transitions over it.

    The parent allocates the state before workers spawn; every worker's coordinator holds it
    and any worker may execute a dissolution.  After the workers finish, `events()` returns the
    dissolution log.
    """

    def __init__(self, context: "BaseContext", n_workers: int, k: int, score_length: int) -> None:
        """Allocate the slots, the identity assignment, and the dissolution log in shared memory.

        Args:
            context: the spawn context whose shared-memory primitives back everything here, so
                workers inherit the state at spawn.
            n_workers: the worker count, fixing the slot count and every table size.
            k: maximum selection size the slots hold.
            score_length: number of components in the workers' score tuples.
        """
        self._n_workers = n_workers
        self._score_length = score_length
        self._slots = [GroupExchangeSlot(context, k=k, score_length=score_length) for _ in range(n_workers)]
        self._assignment = context.Array("i", list(range(n_workers)), lock=False)
        self._n_alive_groups = context.Value("i", n_workers, lock=False)
        self._transition_lock = context.Lock()
        # The dissolution log is preallocated: exactly n_workers - 1 dissolutions can ever
        # happen.  The stored assignment is the post-event table; NaN marks a never-written
        # slot's score.
        n_events = max(n_workers - 1, 1)
        self._ev_count = context.Value("i", 0, lock=False)
        self._ev_fraction = context.Array("d", n_events, lock=False)
        self._ev_dissolved = context.Array("i", n_events, lock=False)
        self._ev_assignment = context.Array("i", n_events * n_workers, lock=False)
        self._ev_scores = context.Array("d", n_events * n_workers * score_length, lock=False)

    def coordinator_for(self, worker_index: int) -> "DynamicGroupCoordinator":
        """Return the given worker's coordinator, bound to this shared state."""
        return DynamicGroupCoordinator(self, worker_index)

    # -------------------------------------------------------------------------
    #  Worker-side operations
    # -------------------------------------------------------------------------
    def maybe_dissolve(self, progress_fraction: float) -> None:
        """Dissolve groups until the alive count matches the schedule at the given fraction.

        The no-transition case — by far the common one — costs a single lock-free read; the
        transition itself runs under the one transition lock, and the count re-check inside it
        means a concurrent caller that lost the race dissolves nothing.
        """
        target = dynamic_group_count(self._n_workers, progress_fraction)
        if self._n_alive_groups.value <= target:
            return
        with self._transition_lock:
            while self._n_alive_groups.value > target:
                self._dissolve_worst(progress_fraction)

    def exchange(
        self, worker_index: int, score: tuple[float, ...], selection: "NDArray[np.int32]"
    ) -> "NDArray[np.int32] | None":
        """Exchange with the worker's currently assigned slot; see `GroupExchangeSlot.exchange`.

        The assignment read is one shared-memory element; re-reading it every exchange keeps the
        workers free of any synchronization with dissolution writes.
        """
        return self._slots[self._assignment[worker_index]].exchange(score, selection)

    def _dissolve_worst(self, progress_fraction: float) -> None:
        """Dissolve the worst-scoring group and reassign its workers to the strongest short groups.

        The caller holds the transition lock; the assignment table is the single source of
        membership, so sizes are counted from it.
        """
        assignment = list(self._assignment)
        alive = sorted(set(assignment))
        scores = {group: self._slots[group].peek_score() for group in alive}
        # a never-written slot ranks below any written one (the empty tuple sorts below any real
        # score tuple); ties go to the lowest group index
        worst = min(alive, key=lambda group: (scores[group] or (), group))
        sizes = {group: assignment.count(group) for group in alive if group != worst}
        reassignments: dict[int, int] = {}
        for worker in [index for index, group in enumerate(assignment) if group == worst]:
            target = self._reassignment_target(scores, sizes)
            sizes[target] += 1
            self._assignment[worker] = target
            reassignments[worker] = target
        self._n_alive_groups.value -= 1
        self._record_event(progress_fraction, worst, scores)

    @staticmethod
    def _reassignment_target(scores: dict[int, tuple[float, ...] | None], sizes: dict[int, int]) -> int:
        """Return the group a freed worker joins: the best-scoring one still short of the largest.

        When every surviving group has the same size, no group is short and the best-scoring one
        overall takes the worker.
        """
        largest = max(sizes.values())
        short = [group for group, size in sizes.items() if size < largest]
        pool = short if short else list(sizes)
        return max(pool, key=lambda group: (scores[group] or (), -group))

    # -------------------------------------------------------------------------
    #  Dissolution log
    # -------------------------------------------------------------------------
    def _record_event(
        self, progress_fraction: float, dissolved: int, scores: dict[int, tuple[float, ...] | None]
    ) -> None:
        """Append one dissolution to the shared log; the caller holds the transition lock."""
        index = self._ev_count.value
        self._ev_fraction[index] = progress_fraction
        self._ev_dissolved[index] = dissolved
        self._ev_assignment[index * self._n_workers : (index + 1) * self._n_workers] = list(self._assignment)
        flat_scores = []
        for group in range(self._n_workers):
            score = scores.get(group)
            flat_scores.extend(score if score is not None else [math.nan] * self._score_length)
        start = index * self._n_workers * self._score_length
        self._ev_scores[start : start + len(flat_scores)] = flat_scores
        self._ev_count.value = index + 1

    def events(self) -> list[DissolutionEvent]:
        """Return the dissolution log as `DissolutionEvent`s; call after the workers finished."""
        events = []
        previous_assignment = list(range(self._n_workers))
        alive = set(range(self._n_workers))
        for index in range(self._ev_count.value):
            assignment = list(self._ev_assignment[index * self._n_workers : (index + 1) * self._n_workers])
            start = index * self._n_workers * self._score_length
            slot_scores: dict[int, tuple[float, ...] | None] = {}
            for group in sorted(alive):
                score = tuple(self._ev_scores[start + group * self._score_length :][: self._score_length])
                slot_scores[group] = None if math.isnan(score[0]) else score
            events.append(
                DissolutionEvent(
                    progress_fraction=self._ev_fraction[index],
                    dissolved_group=self._ev_dissolved[index],
                    slot_scores=slot_scores,
                    reassignments={
                        worker: group for worker, group in enumerate(assignment) if group != previous_assignment[worker]
                    },
                )
            )
            alive.discard(self._ev_dissolved[index])
            previous_assignment = assignment
        return events


class DynamicGroupCoordinator(WorkerCoordinator):
    """A worker of a dynamic solve runs the schedule and exchanges through the shared state."""

    def __init__(self, group_state: DynamicGroupState, worker_index: int) -> None:
        """Bind the coordinator to the solve's shared group state and the worker's index."""
        self._group_state = group_state
        self._worker_index = worker_index

    def at_batch_boundary(self, state: SolverState, progress_fraction: float) -> None:
        """Bring the group count down to the schedule's target, then exchange with the currently assigned slot."""
        self._group_state.maybe_dissolve(progress_fraction)
        incoming = self._group_state.exchange(self._worker_index, state.score.as_tuple(), state.selected_index_array)
        if incoming is not None:
            state.adopt_selection(incoming)
