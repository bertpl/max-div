"""A group-merge schedule maps a worker's progress fraction to the group count it should see.

Every schedule is a fixed function of the progress fraction: it never reads scores or timing, so
each worker can evaluate it on its own and reach the same count.  `WorkerGroupState` compares the
count with the number of alive groups and dissolves groups until they match.
"""

import math
from abc import ABC, abstractmethod

# The exponent the power-law schedule raises the remaining progress to.  Rate 2 gives the
# best-scoring groups extra workers while most of the budget is still ahead — at rate 1 a
# 12-worker solve makes its last merge with a twelfth of the budget left.
DEFAULT_GROUP_MERGE_RATE: float = 2.0
# Below 1 merges slower than the linear schedule, which nothing asks for; at 10 a 12-worker solve
# is a single group from under a quarter of the budget onward.
GROUP_MERGE_RATE_BOUNDS: tuple[float, float] = (1.0, 10.0)


class GroupMergeSchedule(ABC):
    """The group count a parallel solve should have at each progress fraction."""

    @abstractmethod
    def group_count(self, progress_fraction: float) -> int:
        """Return the scheduled group count at the given progress fraction (clamped to 0..1)."""


class FixedGroupCount(GroupMergeSchedule):
    """A fixed grouping's schedule: the configured group count, so no merge ever fires."""

    def __init__(self, n_groups: int) -> None:
        """Hold the configured group count."""
        self._n_groups = n_groups

    def group_count(self, progress_fraction: float) -> int:
        """Return the configured group count, whatever the progress."""
        return self._n_groups


class PowerLawGroupMerge(GroupMergeSchedule):
    """The dynamic grouping's schedule: `n_workers * (1 - progress) ** rate` groups, rounded up.

    Rate 1 decreases the count linearly, so every group count holds an equal share of the budget.
    A larger rate merges sooner: at the start of the solve the count drops `rate` times faster
    than under the linear schedule (the derivative of `(1 - x) ** rate` at 0 is `-rate`).
    """

    def __init__(self, n_workers: int, rate: float) -> None:
        """Hold the starting group count and the exponent; the builder validates the rate's range."""
        self._n_workers = n_workers
        self._rate = rate

    def group_count(self, progress_fraction: float) -> int:
        """Return the scheduled count, at least one.

        The fraction is clamped to 0..1, so a not-yet-started tracker reads as the start and an
        overspent budget as the end.
        """
        fraction = min(max(progress_fraction, 0.0), 1.0)
        return max(1, math.ceil(self._n_workers * (1.0 - fraction) ** self._rate))
