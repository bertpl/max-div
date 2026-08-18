"""A portfolio builder configures and builds several solvers over one problem."""

from collections.abc import Sequence
from typing import Self, cast

from max_div._core._utils import deterministic_hash_int64
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._duration import TargetDuration
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

from ._base import SolverBuilderBase


class ParallelMaxDivSolverBuilder(SolverBuilderBase):
    """A portfolio builder configures several workers over one problem, and the best result wins.

    There is no `with_preset`: a preset belongs to a worker (`WorkerConfig`), and workers may run
    different ones.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, problem: MaxDivProblem) -> None:
        """Configure a portfolio over the given problem.

        Args:
            problem: The MaxDivProblem every worker solves.
        """
        super().__init__(problem)
        self._worker_configs: list[WorkerConfig] = []
        self._group_sizes: list[int] = []
        self._target_duration: TargetDuration | None = None

    # -------------------------------------------------------------------------
    #  Builder API
    # -------------------------------------------------------------------------
    def with_workers(
        self,
        target_duration: TargetDuration,
        workers: int | Sequence[WorkerConfig] | Sequence[Sequence[WorkerConfig]] | None = None,
        n_groups: int | None = None,
    ) -> Self:
        """Set what the portfolio runs, for how long each worker runs it, and how workers form groups.

        The workers run side by side, so the portfolio takes as long as one of them rather than the
        sum.  Presets differ in iteration speed, so when workers run different ones a wall-clock
        budget keeps them to the same real time where an iteration count would not.

        Workers form **worker groups**: within a group, workers adopt the best selection any
        member has found so far; groups never communicate, and the best worker over all groups
        wins.  Groups of one make those workers fully independent — `n_groups` equal to the
        worker count is the fully independent portfolio.

        Args:
            target_duration: the wall-clock budget each worker runs for (see `TargetDuration`).
            workers: an integer runs the default configuration that many times; a flat sequence
                gives one configuration per worker; a nested sequence gives one inner
                sequence per group, fixing both grouping and configurations; omitting it
                uses `default_worker_count()`.
            n_groups: number of groups; only combines with an integer (or omitted) `workers` —
                a nested sequence carries its own grouping, a flat sequence uses the
                default.  Omitting it uses `default_group_count()`.

        Raises:
            ValueError: If `n_groups` accompanies a sequence form, falls outside 1..worker count,
                or the sequence mixes configurations and groups.
        """
        self._target_duration = target_duration
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
            self._worker_configs = [config for group in groups for config in group]
            self._group_sizes = [len(group) for group in groups]
        return self

    # -------------------------------------------------------------------------
    #  Build
    # -------------------------------------------------------------------------
    def build(self) -> ParallelMaxDivSolver:
        """Build the portfolio: one solver configuration per worker over a store they will share.

        Raises:
            ValueError: If no workers were configured.
        """
        if self._target_duration is None or not self._worker_configs:
            raise ValueError("A portfolio needs workers; call with_workers before build.")
        warn_about_worker_count(len(self._worker_configs))
        resolved, label = self._select_storage()
        group_size_per_worker = [size for size in self._group_sizes for _ in range(size)]
        return ParallelMaxDivSolver(
            problem=self._problem,
            storage=resolved,
            worker_configs=self._worker_configs,
            solver_configs=[
                self._solver_config_for(index, worker, self._target_duration, label, group_size_per_worker[index])
                for index, worker in enumerate(self._worker_configs)
            ],
            group_sizes=self._group_sizes,
        )

    def _solver_config_for(
        self, index: int, worker: WorkerConfig, duration: TargetDuration, storage_label: str, group_size: int
    ) -> SolverConfig:
        """Return the solver configuration for one worker.

        Worker seeds are derived from the portfolio seed rather than set, so one seed pins every
        worker's search while the workers still search differently (with cooperating groups the
        run stays timing-dependent, so the seed only makes a fully independent portfolio
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
            solver_steps=[InitializationStep(worker.init_strategy or init_strategy), *optim_steps],
            seed=int(deterministic_hash_int64(("parallel_worker_seed", self._seed, index))),
            constraint_penalty=self._constraint_penalty,
            distance_storage_label=storage_label,
            # tighter batches give cooperative workers faster incumbent exchanges
            batch_seconds=COOPERATIVE_BATCH_SECONDS if group_size > 1 else REPORTING_BATCH_SECONDS,
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
