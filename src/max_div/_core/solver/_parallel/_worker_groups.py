"""Fixed and dynamic worker groups share one framework and differ only in the schedule.

At every point of a solve, each worker is assigned to exactly one group, and at each batch
boundary it exchanges its selection through its group's slot.

- A **fixed** grouping keeps the assignment as configured for the whole solve — its scheduled
  group count is a constant, so no transition ever fires.
- A **dynamic** grouping starts every worker in its own group and decreases the scheduled count
  to one over each worker's progress fraction (see `merge_fractions`).  Each decrease dissolves
  the group whose exchange slot holds the worst score — the group whose best score is lowest so
  far — and reassigns its workers to the strongest groups that are short a member, so they
  reinforce searches that can still win.

The workers themselves run the schedule; no separate process does:

- **`WorkerGroupState` is the shared-memory record of the grouping**: one slot per worker, an
  assignment table mapping each worker to its slot, the alive group count, and the dissolution
  log.
- **`WorkerGroupCoordinator` runs the schedule from inside the workers**: at each batch boundary
  a worker computes the scheduled group count from its own progress fraction, and whichever
  worker first sees the alive count exceed the schedule executes the dissolution itself, under a
  single transition lock.

Ranking groups on their slots makes dissolution safe: a slot only ever accepts a strictly better
selection (`GroupExchangeSlot.exchange`), so it holds its group's best score so far, and
dissolving the worst-slot group can never discard the overall best selection.

A worker learns of its reassignment at its next batch boundary, so membership changes are
eventual, not synchronized — a worker mid-batch keeps exchanging with its old slot for at most
one more batch.
"""

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

from max_div._core.solver._solver_state import SolverState

from ._coordinator import WorkerCoordinator
from ._exchange_slot import GroupExchangeSlot

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    import numpy as np
    from numpy.typing import NDArray


# ==================================================================================================
#  Merge schedule
# ==================================================================================================
# The dynamic schedule raises the remaining progress to this exponent.  Rate 2 gives the
# best-scoring groups extra workers while most of the budget is still ahead — at rate 1 a
# 12-worker solve makes its last merge with a twelfth of the budget left.
DEFAULT_GROUP_MERGE_RATE: float = 2.0
# Below 1 merges slower than the linear schedule, which nothing asks for; at 10 a 12-worker solve
# is a single group from under a quarter of the budget onward.
GROUP_MERGE_RATE_BOUNDS: tuple[float, float] = (1.0, 10.0)


def merge_fractions(n_workers: int, merge_rate: float) -> list[float]:
    """Return the ascending progress fractions at which a dynamic grouping merges, one per merge.

    The scheduled group count at fraction `f` is `ceil(n_workers * (1 - f) ** merge_rate)`, so
    the i-th merge (from `n_workers` groups down to `n_workers - i`) fires where
    `(1 - f) ** merge_rate` drops to `(n_workers - i) / n_workers`.  At rate 1 the merges sit at
    `i / n_workers`, so every group count holds an equal share of the budget.
    """
    return [1.0 - ((n_workers - i) / n_workers) ** (1.0 / merge_rate) for i in range(1, n_workers)]


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


class WorkerGroupState:
    """The shared group state records a parallel solve's grouping and executes the transitions over it.

    The parent allocates the state before workers spawn; every worker's coordinator holds it
    and any worker may execute a dissolution.  After the workers finish, `events()` returns the
    dissolution log — empty for a fixed grouping, whose schedule never fires a transition.
    """

    def __init__(
        self,
        context: "BaseContext",
        group_sizes: list[int],
        k: int,
        score_length: int,
        dynamic: bool,
        merge_rate: float = DEFAULT_GROUP_MERGE_RATE,
    ) -> None:
        """Allocate the slots, the configured assignment, and the dissolution log in shared memory.

        Args:
            context: the spawn context whose shared-memory primitives back everything here, so
                workers inherit the state at spawn.
            group_sizes: the initial grouping, as consecutive run lengths over the worker order;
                a dynamic grouping starts from groups of one.
            k: maximum selection size the slots hold.
            score_length: number of components in the workers' score tuples.
            dynamic: whether the group count follows the dynamic schedule; a fixed grouping's
                scheduled count is the configured group count, so it never changes.
            merge_rate: the dynamic schedule's exponent (see `merge_fractions`); ignored by a
                fixed grouping.  The builder validates the range; direct construction does not.
        """
        # --- configuration ----------------------
        if any(size <= 0 for size in group_sizes):
            # the builder validates this too; the re-check protects direct construction: an
            # empty group makes the assignment table reference slots that were never allocated
            raise ValueError(f"Every worker group needs at least one worker; got sizes {group_sizes}.")
        self._n_workers: int = sum(group_sizes)
        self._initial_group_count: int = len(group_sizes)
        self._dynamic: bool = dynamic
        self._score_length: int = score_length
        self._merge_fractions: list[float] = merge_fractions(self._n_workers, merge_rate) if dynamic else []

        # --- shared grouping state --------------
        # one exchange slot per worker, so the dynamic schedule can start from groups of one
        self._slots = [GroupExchangeSlot(context, k=k, score_length=score_length) for _ in range(self._n_workers)]
        # the assignment table: each worker's current group (= slot index), starting as configured
        initial_assignment = [group for group, size in enumerate(group_sizes) for _ in range(size)]
        self._assignment = context.Array("i", initial_assignment, lock=False)
        # the number of groups that still have workers; read lock-free as the transition guard
        self._n_alive_groups = context.Value("i", self._initial_group_count, lock=False)
        self._transition_lock = context.Lock()

        # --- shared dissolution log -------------
        # The dissolution log is preallocated: exactly n_workers - 1 dissolutions can ever
        # happen.  The stored assignment is the post-event table; NaN marks a never-written
        # slot's score.
        n_events = max(self._n_workers - 1, 1)
        self._ev_count = context.Value("i", 0, lock=False)
        self._ev_fraction = context.Array("d", n_events, lock=False)
        self._ev_dissolved = context.Array("i", n_events, lock=False)
        self._ev_assignment = context.Array("i", n_events * self._n_workers, lock=False)
        self._ev_scores = context.Array("d", n_events * self._n_workers * score_length, lock=False)

    def coordinator_for(self, worker_index: int) -> "WorkerGroupCoordinator":
        """Return the given worker's coordinator, bound to this shared state."""
        return WorkerGroupCoordinator(self, worker_index)

    # -------------------------------------------------------------------------
    #  Worker-side operations
    # -------------------------------------------------------------------------
    def maybe_dissolve(self, progress_fraction: float) -> None:
        """Dissolve groups until the alive count matches the schedule at the given fraction.

        The no-transition case — every boundary of a fixed grouping, and most boundaries of a
        dynamic one — costs a single lock-free read; the transition itself runs under the one
        transition lock, and the count re-check inside it means a concurrent caller that lost
        the race dissolves nothing.
        """
        target = self._scheduled_count(progress_fraction)
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

    def _scheduled_count(self, progress_fraction: float) -> int:
        """Return the group count the schedule asks for at the given progress fraction.

        The count is the starting group count minus the merges scheduled at or below the
        fraction (see `merge_fractions`); a fixed grouping schedules no merges, so its count is
        the configured one.  The fraction is clamped to 0..1, so a not-yet-started tracker reads
        as the start and an overspent budget as the end.
        """
        fraction = min(max(progress_fraction, 0.0), 1.0)
        return max(1, self._initial_group_count - bisect_right(self._merge_fractions, fraction))

    def _dissolve_worst(self, progress_fraction: float) -> None:
        """Dissolve the worst-scoring group and reassign its workers to the strongest short groups.

        The caller holds the transition lock; the assignment table is the single source of
        membership, so sizes are counted from it.
        """
        group_per_worker: list[int] = list(self._assignment)
        alive_groups: list[int] = sorted(set(group_per_worker))
        score_per_group = {group: self._slots[group].peek_score() for group in alive_groups}
        # a never-written slot ranks below any written one (the empty tuple sorts below any real
        # score tuple); ties go to the lowest group index
        worst_group = min(alive_groups, key=lambda group: (score_per_group[group] or (), group))
        size_per_surviving_group = {
            group: group_per_worker.count(group) for group in alive_groups if group != worst_group
        }
        reassignments: dict[int, int] = {}
        # hand each of the dissolved group's workers to a surviving group, one at a time, so
        # every placement sees the sizes the previous one produced
        for worker in [index for index, group in enumerate(group_per_worker) if group == worst_group]:
            target = self._reassignment_target(score_per_group, size_per_surviving_group)
            size_per_surviving_group[target] += 1
            self._assignment[worker] = target
            reassignments[worker] = target
        self._n_alive_groups.value -= 1
        self._record_event(progress_fraction, worst_group, score_per_group)

    @staticmethod
    def _reassignment_target(
        score_per_group: dict[int, tuple[float, ...] | None], size_per_surviving_group: dict[int, int]
    ) -> int:
        """Return the group a freed worker joins: the best-scoring one among the smallest groups.

        Args:
            score_per_group: each group's slot score, None for a never-written slot; among the
                candidate groups, the best score takes the worker.
            size_per_surviving_group: current member count per surviving group; the sizes decide
                which groups are candidates — those of the smallest size.
        """
        smallest = min(size_per_surviving_group.values())
        pool = [group for group, size in size_per_surviving_group.items() if size == smallest]
        return max(pool, key=lambda group: (score_per_group[group] or (), -group))

    # -------------------------------------------------------------------------
    #  Dissolution log
    # -------------------------------------------------------------------------
    def _record_event(
        self, progress_fraction: float, dissolved: int, score_per_group: dict[int, tuple[float, ...] | None]
    ) -> None:
        """Append one dissolution to the shared log; the caller holds the transition lock."""
        index = self._ev_count.value
        self._ev_fraction[index] = progress_fraction
        self._ev_dissolved[index] = dissolved
        self._ev_assignment[index * self._n_workers : (index + 1) * self._n_workers] = list(self._assignment)
        flat_scores = []
        for group in range(self._n_workers):
            score = score_per_group.get(group)
            flat_scores.extend(score if score is not None else [math.nan] * self._score_length)
        start = index * self._n_workers * self._score_length
        self._ev_scores[start : start + len(flat_scores)] = flat_scores
        self._ev_count.value = index + 1

    def events(self) -> list[DissolutionEvent]:
        """Return the dissolution log as `DissolutionEvent`s; call after the workers finished."""
        events = []
        previous_assignment = self._initial_assignment()
        alive = set(previous_assignment)
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

    def _initial_assignment(self) -> list[int]:
        """Return the assignment the solve started from.

        A dynamic grouping starts from the identity assignment; a fixed grouping never moves a
        worker, so its current table is still the initial one.
        """
        if self._dynamic:
            return list(range(self._n_workers))
        return list(self._assignment)


class WorkerGroupCoordinator(WorkerCoordinator):
    """A parallel solve's workers run the grouping schedule and exchange through the shared state."""

    def __init__(self, group_state: WorkerGroupState, worker_index: int) -> None:
        """Bind the coordinator to the solve's shared group state and the worker's index."""
        self._group_state = group_state
        self._worker_index = worker_index

    def at_batch_boundary(self, state: SolverState, progress_fraction: float) -> None:
        """Bring the group count down to the schedule's target, then exchange with the currently assigned slot."""
        self._group_state.maybe_dissolve(progress_fraction)
        incoming = self._group_state.exchange(self._worker_index, state.score.as_tuple(), state.selected_index_array)
        if incoming is not None:
            state.adopt_selection(incoming)
