"""Several worker processes — a portfolio — solve one problem at once, and the best result wins.

Every worker reads the same distances and runs its own search.  The search varies between workers;
the objective does not, because comparing what they found needs one answer to which selection is
better.
"""

from ._coordinator import CooperativeCoordinator, IndependentCoordinator, WorkerCoordinator
from ._executor import run_portfolio, solve_in_worker
from ._incumbent_slot import GroupIncumbentSlot
from ._result import WorkerResult, best_result
from ._solution import ParallelMaxDivSolution, WorkerSummary
from ._solver import ParallelMaxDivSolver, default_worker_count, warn_about_worker_count
from ._worker_config import WorkerConfig

__all__ = [
    "CooperativeCoordinator",
    "GroupIncumbentSlot",
    "IndependentCoordinator",
    "ParallelMaxDivSolution",
    "ParallelMaxDivSolver",
    "WorkerConfig",
    "WorkerCoordinator",
    "WorkerResult",
    "WorkerSummary",
    "best_result",
    "default_worker_count",
    "run_portfolio",
    "solve_in_worker",
    "warn_about_worker_count",
]
