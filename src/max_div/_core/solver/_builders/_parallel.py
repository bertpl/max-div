"""A portfolio builder configures and builds several solvers over one problem."""

from collections.abc import Sequence
from typing import Self

from max_div._core._utils import deterministic_hash_int64
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._duration import TargetDuration
from max_div._core.solver._parallel import ParallelMaxDivSolver, WorkerConfig, warn_about_worker_count
from max_div._core.solver._presets import get_preset_strategies
from max_div._core.solver._solver_config import SolverConfig
from max_div._core.solver._solver_step import InitializationStep

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

        :param problem: The MaxDivProblem every worker solves.
        """
        super().__init__(problem)
        self._worker_configs: list[WorkerConfig] = []
        self._target_duration: TargetDuration | None = None

    # -------------------------------------------------------------------------
    #  Builder API
    # -------------------------------------------------------------------------
    def with_workers(self, target_duration: TargetDuration, workers: int | Sequence[WorkerConfig]) -> Self:
        """Set what the portfolio runs, and for how long each worker runs it.

        The workers run side by side, so the portfolio takes as long as one of them rather than the
        sum.  Presets differ in iteration speed, so when workers run different ones a wall-clock
        budget keeps them to the same real time where an iteration count would not.

        :param workers: an integer runs the default configuration that many times; a sequence gives
                        one configuration per worker.
        """
        self._target_duration = target_duration
        self._worker_configs = [WorkerConfig() for _ in range(workers)] if isinstance(workers, int) else list(workers)
        return self

    # -------------------------------------------------------------------------
    #  Build
    # -------------------------------------------------------------------------
    def build(self) -> ParallelMaxDivSolver:
        """Build the portfolio: one solver configuration per worker over a store they will share.

        :raises ValueError: If no workers were configured.
        """
        if self._target_duration is None or not self._worker_configs:
            raise ValueError("A portfolio needs workers; call with_workers before build.")
        warn_about_worker_count(len(self._worker_configs))
        resolved, label = self._select_storage()
        return ParallelMaxDivSolver(
            problem=self._problem,
            storage=resolved,
            worker_configs=self._worker_configs,
            solver_configs=[
                self._solver_config_for(index, worker, self._target_duration, label)
                for index, worker in enumerate(self._worker_configs)
            ],
        )

    def _solver_config_for(
        self, index: int, worker: WorkerConfig, duration: TargetDuration, storage_label: str
    ) -> SolverConfig:
        """Return the solver configuration for one worker.

        Worker seeds are derived from the portfolio seed rather than set, so one seed reproduces the
        whole portfolio while the workers still search differently.  The derivation reduces to an
        int64 because the seed is reported back for replaying a worker on its own, and salts the
        tuple so a worker seed cannot coincide with a seed derived the same way elsewhere.
        """
        init_strategy, optim_steps = get_preset_strategies(worker.preset, duration)
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
        )
