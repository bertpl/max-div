"""A parallel-solver builder configures and builds several solvers over one problem."""

from collections.abc import Sequence
from typing import Self, cast

from max_div._core._utils import deterministic_hash_int64
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._duration import E2eBudget, TargetDuration
from max_div._core.solver._parallel import (
    ParallelMaxDivSolver,
    WorkerConfig,
    default_group_count,
    default_worker_count,
    warn_about_worker_count,
)
from max_div._core.solver._presets import get_preset_strategies
from max_div._core.solver._solver_config import SolverConfig
from max_div._core.solver._solver_step import COOPERATIVE_BATCH_SECONDS, REPORTING_BATCH_SECONDS, InitializationStep
from max_div._core.solver._strategies import InitializationStrategy

from ._base import SolverBuilderBase


class ParallelMaxDivSolverBuilder(SolverBuilderBase):
    """A parallel-solver builder configures several workers over one problem, and the best result wins.

    There is no `with_preset`: a preset belongs to a worker (`WorkerConfig`), and workers may run
    different ones.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, problem: MaxDivProblem) -> None:
        """Configure a parallel solve over the given problem.

        Args:
            problem: The MaxDivProblem every worker solves.
        """
        super().__init__(problem)
        self._worker_configs: list[WorkerConfig] = []
        self._group_sizes: list[int] = []
        self._target_duration: TargetDuration | None = None
        self._dynamic_groups: bool = False
        self._init_strategy: InitializationStrategy | None = None

    # -------------------------------------------------------------------------
    #  Builder API
    # -------------------------------------------------------------------------
    def with_workers(self, target_duration: TargetDuration, n_workers: int | None = None) -> Self:
        """Set `n_workers` default workers with the **dynamic** grouping.

        The workers run side by side and the best result wins; their grouping consolidates from
        one group per worker toward a single all-worker group (see `_worker_groups`).
        `with_custom_worker_groups` is the alternative for a fixed grouping or per-worker
        configurations.

        Args:
            target_duration: the budget each worker runs for (see `TargetDuration`).
            n_workers: how many workers solve; omitting it uses `default_worker_count()`.
        """
        self._target_duration = target_duration
        n = default_worker_count() if n_workers is None else n_workers
        self._worker_configs = [WorkerConfig() for _ in range(n)]
        self._group_sizes = [1] * n
        self._dynamic_groups = True
        return self

    def set_initialization_strategy(self, init_strategy: InitializationStrategy) -> Self:
        """Replace every worker's preset initialization with the given strategy.

        A worker whose `WorkerConfig.init_strategy` is set keeps that — the per-worker choice is
        the more specific one.  Every worker still gets its own derived seed, so a randomized
        strategy varies per worker as it would under the preset initialization.
        """
        self._init_strategy = init_strategy
        return self

    def with_custom_worker_groups(
        self,
        target_duration: TargetDuration,
        workers: int | Sequence[WorkerConfig] | Sequence[Sequence[WorkerConfig]] | None = None,
        n_groups: int | None = None,
    ) -> Self:
        """Set custom worker configurations and a **fixed** grouping, kept for the whole solve.

        Presets differ in iteration speed, so when workers run different ones a wall-clock
        budget keeps them to the same real time where an iteration count would not.  Groups of
        one make workers fully independent — `n_groups` equal to the worker count puts every
        worker in a group of one.

        Args:
            target_duration: the budget each worker runs for (see `TargetDuration`).
            workers: an integer runs the default configuration that many times; a flat sequence
                gives one configuration per worker; a nested sequence gives one inner
                sequence per group, fixing both grouping and configurations; omitting it
                uses `default_worker_count()`.
            n_groups: number of groups; only combines with an integer (or omitted) `workers` —
                a nested sequence carries its own grouping.  Omitting it uses
                `default_group_count()`.

        Raises:
            ValueError: If `n_groups` accompanies a sequence form, falls outside 1..worker count,
                or the sequence mixes configurations and groups.
        """
        self._target_duration = target_duration
        self._dynamic_groups = False
        if workers is None:
            workers = default_worker_count()
        if isinstance(workers, int):
            self._worker_configs = [WorkerConfig() for _ in range(workers)]
            self._group_sizes = _resolve_group_sizes(workers, n_groups)
        elif any(isinstance(entry, WorkerConfig) for entry in workers):
            if not all(isinstance(entry, WorkerConfig) for entry in workers):
                raise ValueError("Workers must be all configurations (flat) or all groups (nested), not a mix.")
            if n_groups is not None:
                raise ValueError(
                    "n_groups only combines with an integer worker count; a flat sequence uses the default grouping."
                )
            self._worker_configs = [cast("WorkerConfig", entry) for entry in workers]
            self._group_sizes = _resolve_group_sizes(len(self._worker_configs), None)
        else:
            if n_groups is not None:
                raise ValueError(
                    "n_groups only combines with an integer worker count; a nested sequence carries its own grouping."
                )
            groups = [list(cast("Sequence[WorkerConfig]", group)) for group in workers]
            if any(not group for group in groups):
                # the empty group is rejected here so the error names the user's input; the
                # WorkerGroupState re-check documents the assignment-table mechanism it breaks
                raise ValueError("Worker groups must not be empty; drop the empty inner sequence(s).")
            self._worker_configs = [config for group in groups for config in group]
            self._group_sizes = [len(group) for group in groups]
        return self

    # -------------------------------------------------------------------------
    #  Build
    # -------------------------------------------------------------------------
    def build(self) -> ParallelMaxDivSolver:
        """Build the parallel solver: one solver configuration per worker over a store they will share.

        Raises:
            ValueError: If no workers were configured.
        """
        if self._target_duration is None or not self._worker_configs:
            raise ValueError("A parallel solver needs workers; call with_workers or with_custom_worker_groups first.")
        warn_about_worker_count(len(self._worker_configs))
        if self._init_strategy is not None:
            # Validate here so an unsupported metric fails at build, not first inside a spawned worker.
            self._init_strategy.validate_diversity_metric(self._diversity_metric)
        resolved, label = self._select_storage()
        e2e_budget = self._resolve_e2e_budget()
        batch_intervals = self._batch_interval_per_worker()
        return ParallelMaxDivSolver(
            problem=self._problem,
            storage=resolved,
            worker_configs=self._worker_configs,
            solver_configs=[
                self._solver_config_for(index, worker, self._target_duration, label, batch_intervals[index], e2e_budget)
                for index, worker in enumerate(self._worker_configs)
            ],
            group_sizes=self._group_sizes,
            dynamic_groups=self._dynamic_groups,
        )

    def _batch_interval_per_worker(self) -> list[float]:
        """Return each worker's batch interval: tight for workers that can share a group, coarse for lone ones.

        Under the dynamic grouping every worker can end up sharing a group.  Under a fixed
        grouping only the members of groups of two or more can, and a permanently lone worker
        keeps the coarser reporting interval.
        """
        if self._dynamic_groups:
            return [COOPERATIVE_BATCH_SECONDS] * len(self._worker_configs)
        return [
            COOPERATIVE_BATCH_SECONDS if size > 1 else REPORTING_BATCH_SECONDS
            for size in self._group_sizes
            for _ in range(size)
        ]

    def _solver_config_for(
        self,
        index: int,
        worker: WorkerConfig,
        duration: TargetDuration,
        storage_label: str,
        batch_seconds: float,
        e2e_budget: "E2eBudget | None",
    ) -> SolverConfig:
        """Return the solver configuration for one worker.

        Worker seeds are derived from the parallel solver's seed rather than set, so one seed pins every
        worker's search while the workers still search differently (with cooperating groups the
        run stays timing-dependent, so the seed only makes a fully independent set of workers
        reproducible).  The derivation reduces to an int64 because the seed is reported back for
        replaying a worker on its own, and salts the tuple so a worker seed cannot coincide with a
        seed derived the same way elsewhere.
        """
        init_strategy, optim_steps = get_preset_strategies(
            worker.preset, duration, has_constraints=bool(self._constraints)
        )
        return SolverConfig(
            n=self._n,
            k=self._k,
            diversity_metric=self._diversity_metric,
            diversity_tie_breakers=self._determine_diversity_tie_breakers(),
            constraints=self._constraints,
            solver_steps=[
                InitializationStep(worker.init_strategy or self._init_strategy or init_strategy),
                *optim_steps,
            ],
            seed=int(deterministic_hash_int64(("parallel_worker_seed", self._seed, index))),
            constraint_penalty=self._constraint_penalty,
            distance_storage_label=storage_label,
            batch_seconds=batch_seconds,
            e2e_budget=e2e_budget,
        )


def _resolve_group_sizes(n_workers_total: int, n_groups: int | None) -> list[int]:
    """Return the group sizes for a worker total, splitting any remainder over the first groups.

    Args:
        n_workers_total: the number of workers to split into groups.
        n_groups: number of groups; `default_group_count(n_workers_total)` if None.

    Raises:
        ValueError: If `n_groups` falls outside 1..`n_workers_total`.
    """
    if n_groups is None:
        n_groups = default_group_count(n_workers_total)
    elif not 1 <= n_groups <= n_workers_total:
        raise ValueError(f"n_groups must be between 1 and the worker count ({n_workers_total}); got {n_groups}.")
    base, remainder = divmod(n_workers_total, n_groups)
    return [base + 1 if group < remainder else base for group in range(n_groups)]
