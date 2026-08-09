"""Several worker processes solve one problem at once, and the best result they find wins.

Every worker reads the same distances out of shared memory and runs its own search over them, so
what varies between workers is the search, never the objective.  The pieces:

  - `_coordinator` is what a worker calls at each batch boundary.
  - `_result` is what a worker sends back, and the rule that picks a winner.
  - `_executor` starts the workers, collects what they report, and shuts them down.
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
