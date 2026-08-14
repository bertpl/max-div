"""A portfolio solver runs several workers over one shared store and returns the best result they reach."""

import math
import multiprocessing
import os
import warnings
from dataclasses import fields

from max_div._core._warnings import ParallelSolvingWarning
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._distance_storage import DistanceStorage, build_shared_distance_store
from max_div._core.solver._progress_reporting import ProgressReporter, Verbosity
from max_div._core.solver._solution import MaxDivSolution
from max_div._core.solver._solver_config import SolverConfig

from ._coordinator import CooperativeCoordinator, IndependentCoordinator, WorkerCoordinator
from ._executor import run_portfolio
from ._incumbent_slot import GroupIncumbentSlot
from ._result import best_result
from ._solution import ParallelMaxDivSolution, WorkerSummary
from ._worker_config import WorkerConfig


class ParallelMaxDivSolver:
    """A portfolio solver solves one problem with several workers at once, returning the best selection.

    `ParallelMaxDivSolverBuilder` builds this solver, and is where the workers and the shared
    settings are configured.
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
        group_sizes: list[int],
    ) -> None:
        """Hold the problem, the resolved backend, and one configuration per worker.

        :param storage: the already-resolved backend the shared store is built in.
        :param worker_configs: what each worker runs, reported back in the solution.
        :param solver_configs: the solver each worker assembles, in the same order.
        :param group_sizes: how the workers group into groups, as consecutive run lengths over
                             the worker order; sizes must sum to the worker count.
        """
        self._problem = problem
        self._storage = storage
        self._worker_configs = worker_configs
        self._solver_configs = solver_configs
        self._group_sizes = group_sizes

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def solve(self, verbosity: int | Verbosity = Verbosity.TABULAR) -> ParallelMaxDivSolution:
        """Run every worker over one shared store and return the best result, with every worker summarized.

        The distances are built once, into shared memory, and released when the last worker is done.

        :param verbosity: (int | Verbosity) The verbosity level, with the same levels as a single
                          solve (see `Verbosity`), rendered as one combined live view over all
                          workers (see `ParallelProgressView`). The default differs from a single
                          solve's progress bar because parallel runs are typically longer.
        :raises ValueError: If no worker reported a result, which means every one of them failed.
        """
        progress_reporter = ProgressReporter.from_verbosity(verbosity, worker_columns=True)
        with build_shared_distance_store(self._problem, self._storage) as shared_distance_store:
            results = run_portfolio(
                self._solver_configs,
                shared_distance_store.spec,
                self._build_coordinators(),
                progress_reporter=progress_reporter,
            )
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

    def _build_coordinators(self) -> list[WorkerCoordinator]:
        """Return one coordinator per worker: an group's members share a slot, lone workers share nothing."""
        config = self._solver_configs[0]
        context = multiprocessing.get_context("spawn")
        coordinators: list[WorkerCoordinator] = []
        for size in self._group_sizes:
            if size == 1:
                coordinators.append(IndependentCoordinator())
            else:
                # the score length is the three fixed components plus one per tie-breaker (Score.as_tuple)
                slot = GroupIncumbentSlot(context, k=config.k, score_length=3 + len(config.diversity_tie_breakers))
                coordinators.extend([CooperativeCoordinator(slot)] * size)
        return coordinators


def warn_about_worker_count(n_workers: int) -> None:
    """Warn when there are too few workers to help, or more workers than cores."""
    # stacklevel 3 points at the caller of build, two frames up
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


def default_worker_count() -> int:
    """Return the default portfolio size when the caller names none: 3/4 of the logical cores, at least 2.

    The default portfolio is cooperative, and cooperation converts extra workers into shared
    search progress, where extra independent workers only buy insurance against unlucky seeds —
    so more cores in use keep paying off, and the default takes 3/4 where a purely independent
    portfolio took the conservative half.  An explicit count on `with_workers` overrides it.
    """
    return max(2, (os.cpu_count() or 2) * 3 // 4)


def default_group_count(n_workers_total: int) -> int:
    """Return the default group count for a worker total, when the caller names none.

    The default targets roughly twice as many groups as workers per group: the final result is a
    best over all workers, and only groups are independent draws of that best, so groups buy more
    than group size does.  Every group keeps at least 2 workers (a lone worker forms an group of
    one) — on small totals the count collapses to a single group rather than to groups of one,
    preferring cooperation over independence.  An explicit `n_groups` on `with_workers` overrides it.
    """
    return max(1, min(round(math.sqrt(2 * n_workers_total)), n_workers_total // 2))
