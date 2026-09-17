"""The parent assembles, from every worker's own changes, the history of the worker groups over a parallel solve.

Each worker returns the worker group changes it executed, with `elapsed` counted from its own
start. The parent places them on the axis that the solution's checkpoints use, whose zero is the
earliest worker start (see `_trajectory`), and orders them as they happened.

Every change lowers the alive group count by one, so the alive count orders the changes without a
timestamp: two changes executed by different workers moments apart order correctly even when their
workers' elapsed readings, taken before the transition lock, do not.
"""

from dataclasses import replace

from ._result import WorkerResult
from ._worker_group_change import WorkerGroupChange


def worker_group_history(results: list[WorkerResult]) -> list[WorkerGroupChange]:
    """Return every worker's changes with `elapsed` counted from the earliest worker start, in the order they happened.

    Args:
        results: what each worker reported; every result's `t_start` places its changes. Non-empty.
    """
    t_first_start = WorkerResult.earliest_start_time(results)
    merged_group_changes = [
        replace(change, elapsed=result.place_on_shared_axis(change.elapsed, t_first_start))
        for result in results
        for change in result.worker_group_changes
    ]
    return sorted(merged_group_changes, key=lambda change: -change.n_alive_groups_after)
