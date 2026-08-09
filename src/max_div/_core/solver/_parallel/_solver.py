"""A portfolio solver runs several workers over one shared store and returns the best result they reach."""

import os
import warnings
from dataclasses import fields

from max_div._core._warnings import ParallelSolvingWarning
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._distance_storage import DistanceStorage, build_shared_distance_store
from max_div._core.solver._solution import MaxDivSolution
from max_div._core.solver._solver_config import SolverConfig

from ._coordinator import IndependentCoordinator
from ._executor import run_portfolio
from ._result import best_result
from ._solution import ParallelMaxDivSolution, WorkerSummary
from ._worker_config import WorkerConfig


class ParallelMaxDivSolver:
    """A portfolio solver solves one problem with several workers at once, returning the best selection.

    Built by `ParallelMaxDivSolverBuilder`, which is where the workers and the shared settings are
    configured.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        problem: MaxDivProblem,
        storage: DistanceStorage,
        worker_configs: list[WorkerConfig],
        solver_configs: list[SolverConfig],
    ) -> None:
        """Hold the problem, the resolved backend, and one configuration per worker.

        :param problem: the problem every worker solves.
        :param storage: the already-resolved backend the shared store is built in.
        :param worker_configs: what each worker runs, reported back in the solution.
        :param solver_configs: the solver each worker assembles, in the same order.
        """
        self._problem = problem
        self._storage = storage
        self._worker_configs = worker_configs
        self._solver_configs = solver_configs

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def solve(self) -> ParallelMaxDivSolution:
        """Run every worker over one shared store and return the best result, with all of them summarized.

        The distances are built once, into shared memory, and released when the last worker is done.

        :raises ValueError: If no worker reported a result, which means every one of them failed.
        """
        with build_shared_distance_store(self._problem, self._storage) as shared:
            results = run_portfolio(self._solver_configs, shared.spec, IndependentCoordinator())
        winner = best_result(results)
        summaries = [
            WorkerSummary(
                worker_index=result.worker_index,
                config=self._worker_configs[result.worker_index],
                seed=result.seed,
                score=result.score,
                elapsed=result.elapsed,
                has_best_score=result.score == winner.score,
            )
            for result in results
        ]
        inherited = {field.name: getattr(winner.solution, field.name) for field in fields(MaxDivSolution)}
        return ParallelMaxDivSolution(**inherited, workers=summaries, winning_worker=winner.worker_index)


def warn_about_worker_count(n_workers: int) -> None:
    """Warn when there are too few workers to help, or more workers than cores."""
    if n_workers < 2:
        warnings.warn(
            f"A portfolio of {n_workers} worker cannot do better than solving once; "
            "use MaxDivSolverBuilder for a single solve, or configure more workers.",
            ParallelSolvingWarning,
            stacklevel=3,
        )
        return
    n_cores = os.cpu_count() or 1
    if n_workers > n_cores:
        warnings.warn(
            f"A portfolio of {n_workers} workers on {n_cores} cores makes the workers share cores, "
            "so each searches less than it would with fewer of them.",
            ParallelSolvingWarning,
            stacklevel=3,
        )
