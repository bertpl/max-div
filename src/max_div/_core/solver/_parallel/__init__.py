"""Several worker processes solve one problem at once, and the best result wins.

Every worker reads the same distances and runs its own search.  The search varies between workers;
the objective does not, because comparing what they found needs one answer to which selection is
better.
"""

from ._coordinator import WorkerCoordinator
from ._exchange_slot import GroupExchangeSlot
from ._executor import run_workers, solve_in_worker
from ._result import WorkerResult, best_result
from ._solution import ParallelMaxDivSolution, WorkerSummary
from ._solver import ParallelMaxDivSolver, default_group_count, default_worker_count, warn_about_worker_count
from ._worker_config import WorkerConfig
from ._worker_groups import (
    DEFAULT_GROUP_MERGE_RATE,
    GROUP_MERGE_RATE_BOUNDS,
    DissolutionEvent,
    WorkerGroupCoordinator,
    WorkerGroupState,
    merge_fractions,
)

__all__ = [
    "DEFAULT_GROUP_MERGE_RATE",
    "GROUP_MERGE_RATE_BOUNDS",
    "DissolutionEvent",
    "GroupExchangeSlot",
    "ParallelMaxDivSolution",
    "ParallelMaxDivSolver",
    "WorkerConfig",
    "WorkerCoordinator",
    "WorkerGroupCoordinator",
    "WorkerGroupState",
    "WorkerResult",
    "WorkerSummary",
    "best_result",
    "default_group_count",
    "default_worker_count",
    "merge_fractions",
    "run_workers",
    "solve_in_worker",
    "warn_about_worker_count",
]
