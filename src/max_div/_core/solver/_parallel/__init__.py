"""Several worker processes — a portfolio — solve one problem at once, and the best result wins.

Every worker reads the same distances out of shared memory and runs its own search over them, so
what varies between workers is the search, never the objective.
"""

from ._coordinator import IndependentCoordinator, WorkerCoordinator
from ._executor import run_portfolio, solve_in_worker
from ._result import WorkerResult, best_result

__all__ = [
    "IndependentCoordinator",
    "WorkerCoordinator",
    "WorkerResult",
    "best_result",
    "run_portfolio",
    "solve_in_worker",
]
