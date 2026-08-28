"""Several worker processes solve one problem at once, and the best result wins.

Every worker reads the same distances and runs its own search.  The search varies between workers;
the objective does not, because comparing what they found needs one answer to which selection is
better.
"""

from ._adaptive_groups import (
    AdaptiveGroupCoordinator,
    AdaptiveGroupOrchestrator,
    DissolutionEvent,
    adaptive_group_count,
)
from ._coordinator import CooperativeCoordinator, IndependentCoordinator, WorkerCoordinator
from ._executor import run_workers, solve_in_worker
from ._incumbent_slot import GroupIncumbentSlot
from ._result import WorkerResult, best_result
from ._solution import ParallelMaxDivSolution, WorkerSummary
from ._solver import ParallelMaxDivSolver, default_group_count, default_worker_count, warn_about_worker_count
from ._worker_config import WorkerConfig

__all__ = [
    "AdaptiveGroupCoordinator",
    "AdaptiveGroupOrchestrator",
    "CooperativeCoordinator",
    "DissolutionEvent",
    "GroupIncumbentSlot",
    "IndependentCoordinator",
    "ParallelMaxDivSolution",
    "ParallelMaxDivSolver",
    "WorkerConfig",
    "WorkerCoordinator",
    "WorkerResult",
    "WorkerSummary",
    "adaptive_group_count",
    "best_result",
    "default_group_count",
    "default_worker_count",
    "run_workers",
    "solve_in_worker",
    "warn_about_worker_count",
]
