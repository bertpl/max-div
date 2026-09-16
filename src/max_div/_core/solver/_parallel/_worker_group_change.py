"""A worker group change records one dissolution of a worker group during a dynamic parallel solve."""

from dataclasses import dataclass

from max_div._core.solver._duration import Elapsed


@dataclass(frozen=True)
class WorkerGroupChange:
    """A worker group change is one dissolution: the worst-scoring group dissolved and its workers reassigned.

    The worker that executed the change records it; on a parallel solution the changes of all
    workers are listed in the order they happened, and `elapsed` counts from the earliest worker
    start, like the solution's checkpoints, so a change can be placed next to them.

    Args:
        executed_by: index of the worker that executed the change.
        elapsed: when the change happened; the iteration count is the executing worker's own.
        progress_fraction: the executing worker's progress through its optimization step, 0 to 1.
        dissolved_group: index of the dissolved group's exchange slot.
        n_alive_groups_after: how many groups still had workers after the change.
        slot_scores: every then-alive group's slot score, None for a group whose slot was never written.
        reassignments: target group per worker of the dissolved group.
    """

    executed_by: int
    elapsed: Elapsed
    progress_fraction: float
    dissolved_group: int
    n_alive_groups_after: int
    slot_scores: dict[int, tuple[float, ...] | None]
    reassignments: dict[int, int]
