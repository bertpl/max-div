"""The parallel solver runs several workers over one shared store and returns the best result they reach."""

import multiprocessing
import os
import threading
import warnings
from dataclasses import fields

from max_div._core._warnings import ParallelSolvingWarning
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._distance_storage import DistanceStorage, build_shared_distance_store
from max_div._core.solver._duration import E2eBudget
from max_div._core.solver._progress_reporting import ProgressReporter, Verbosity
from max_div._core.solver._solution import MaxDivSolution
from max_div._core.solver._solver_config import SolverConfig

from ._adaptive_groups import AdaptiveGroupOrchestrator, DissolutionEvent
from ._coordinator import CooperativeCoordinator, IndependentCoordinator, WorkerCoordinator
from ._executor import run_workers
from ._incumbent_slot import GroupIncumbentSlot
from ._result import best_result
from ._solution import ParallelMaxDivSolution, WorkerSummary
from ._worker_config import WorkerConfig


class ParallelMaxDivSolver:
    """A parallel solver solves one problem with several workers at once, returning the best selection.

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
        adaptive_groups: bool = False,
        schedule_budget_sec: float | None = None,
    ) -> None:
        """Hold the problem, the resolved backend, and one configuration per worker.

        Args:
            problem: the MaxDivProblem every worker solves.
            storage: the already-resolved backend the shared store is built in.
            worker_configs: what each worker runs, reported back in the solution.
            solver_configs: the solver each worker assembles, in the same order.
            group_sizes: how the workers split into groups, as consecutive run lengths over
                the worker order; sizes must sum to the worker count.
            adaptive_groups: when True, `group_sizes` is ignored and the workers regroup during
                the solve per the schedule in `_adaptive_groups`.
            schedule_budget_sec: wall-clock seconds the adaptive schedule spans when the solver
                configurations carry no end-to-end budget — the per-step time
                budget, counted from the parent's solve start.  With an
                end-to-end budget the schedule reads that budget instead.
        """
        self._problem = problem
        self._storage = storage
        self._worker_configs = worker_configs
        self._solver_configs = solver_configs
        self._group_sizes = group_sizes
        self._adaptive_groups = adaptive_groups
        self._schedule_budget_sec = schedule_budget_sec
        # `last_adaptive_events` holds the most recent adaptive solve's dissolutions, for inspection
        self.last_adaptive_events: list[DissolutionEvent] = []

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def solve(self, verbosity: int | Verbosity = Verbosity.TABULAR) -> ParallelMaxDivSolution:
        """Run every worker over one shared store and return the best result, with every worker summarized.

        The distances are built once, into shared memory, and released when the last worker is done.

        Args:
            verbosity: (int | Verbosity) The verbosity level, with the same levels as a single
                solve (see `Verbosity`), rendered as one combined live view over all
                workers (see `ParallelProgressView`). The default differs from a single
                solve's progress bar because parallel runs are typically longer.

        Raises:
            ValueError: If no worker reported a result, which means every one of them failed.
        """
        progress_reporter = ProgressReporter.from_verbosity(verbosity, worker_columns=True)
        # The budget starts counting before the distances are computed, so it charges the whole
        # setup; the workers receive the started copy and read it against their own (machine-wide)
        # clock.
        solver_configs = self._solver_configs
        e2e_budget = None
        if solver_configs[0].e2e_budget is not None:
            e2e_budget = solver_configs[0].e2e_budget.started()
            solver_configs = [config.with_e2e_budget(e2e_budget) for config in solver_configs]
        orchestrator, orchestrator_thread, stop_orchestrator = None, None, threading.Event()
        if self._adaptive_groups:
            # the schedule reads the e2e budget where one is set; otherwise it spans the per-step
            # time budget from the parent's solve start — a close-enough clock, since the schedule
            # needs only approximate alignment with the workers' own optimization clocks
            if e2e_budget is not None:
                schedule_budget = e2e_budget
            elif self._schedule_budget_sec is not None:
                schedule_budget = E2eBudget(self._schedule_budget_sec).started()
            else:
                raise ValueError("Adaptive worker groups schedule on wall-clock progress; no time budget is set.")
            orchestrator = self._build_orchestrator()
            coordinators: list[WorkerCoordinator] = [
                orchestrator.coordinator_for(index) for index in range(len(solver_configs))
            ]
            orchestrator_thread = threading.Thread(
                target=orchestrator.run, args=(schedule_budget, stop_orchestrator), daemon=True
            )
        else:
            coordinators = self._build_coordinators()
        with build_shared_distance_store(self._problem, self._storage) as shared_distance_store:
            if orchestrator_thread is not None:
                orchestrator_thread.start()
            try:
                results = run_workers(
                    solver_configs,
                    shared_distance_store.spec,
                    coordinators,
                    progress_reporter=progress_reporter,
                )
            finally:
                if orchestrator_thread is not None:
                    stop_orchestrator.set()
                    orchestrator_thread.join(timeout=5.0)
        if orchestrator is not None:
            self.last_adaptive_events = orchestrator.events
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

    def _build_orchestrator(self) -> AdaptiveGroupOrchestrator:
        """Return the orchestrator of an adaptive solve, with one slot per worker."""
        config = self._solver_configs[0]
        context = multiprocessing.get_context("spawn")
        return AdaptiveGroupOrchestrator(
            context,
            n_workers=len(self._solver_configs),
            k=config.k,
            score_length=3 + len(config.diversity_tie_breakers),
        )

    def _build_coordinators(self) -> list[WorkerCoordinator]:
        """Return one coordinator per worker: a worker group's members share a slot, lone workers share nothing."""
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
            f"A parallel solve with {n_workers} worker cannot do better than solving once; "
            "use MaxDivSolverBuilder for a single solve, or configure more workers.",
            ParallelSolvingWarning,
            stacklevel=3,
        )
        return
    n_cores = os.cpu_count() or 1
    if n_workers > n_cores:
        warnings.warn(
            f"A parallel solve with {n_workers} workers on {n_cores} cores makes the workers share cores, "
            "so each searches less than it would with fewer of them.",
            ParallelSolvingWarning,
            stacklevel=3,
        )


def default_worker_count() -> int:
    """Return the default worker count when the caller names none.

    The default configuration is cooperative, and cooperation converts extra workers into shared
    search progress — so more cores in use keep paying off, and the default takes 3/4 of the
    logical cores where purely independent workers would justify only half.
    """
    return max(2, (os.cpu_count() or 2) * 3 // 4)


def default_group_count(n_workers_total: int) -> int:
    """Return the fixed group count used where the adaptive schedule does not apply: nearest a quarter of the total.

    Groups of about four workers matched one all-worker group's result quality in benchmarks while
    spreading the risk of a bad seed over several independent groups.  Rounding to the nearest count keeps
    every group's size between 3 and 5; five workers or fewer form a single group.  This fixed default
    applies wherever the builder resolves the grouping to fixed without an explicit grouping given
    (see `ParallelMaxDivSolverBuilder._resolve_adaptive`).
    """
    return max(1, (n_workers_total + 2) // 4)
