import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from max_div._core._utils._timer import Timer
from max_div._core._warnings import SolverBudgetWarning
from max_div._core.solver._strategies import InitializationStrategy, OptimizationStrategy

from ._duration import E2eBudget, Elapsed, Progress, TargetDuration
from ._progress_reporting import ProgressReporter, SilentProgressReporter
from ._score_checkpoint import ScoreCheckpoint
from ._solver_state import SolverState
from ._step_identity import SolverStepIdentity
from ._strategies._base import StrategyBase

if TYPE_CHECKING:
    from max_div._core.metrics import DiversityObjective

    from ._parallel import WorkerCoordinator

# A caller passes one of these wall-clock targets into `run` as the size of one optimization
# batch.  The default is sized so progress reports can fire ~2x per second; the cooperative value
# gives a group's workers faster exchanges of their shared best, and must stay below the default, or it
# would coarsen reporting instead of tightening exchanges.
REPORTING_BATCH_SECONDS = 0.5
COOPERATIVE_BATCH_SECONDS = 0.05


# =================================================================================================
#  SolverStepResult
# =================================================================================================
@dataclass
class SolverStepResult:
    """The checkpoints of how the score evolved during one step; the last one is the state after its final iteration.

    A step counts `elapsed` from its own start.
    """

    score_checkpoints: list[ScoreCheckpoint]

    @property
    def elapsed(self) -> Elapsed:
        return self.score_checkpoints[-1].elapsed


# =================================================================================================
#  SolverStep
# =================================================================================================
class SolverStep[S: StrategyBase](ABC):
    def __init__(self, strategy: S) -> None:
        self._strategy: S = strategy

    def name(self) -> str:
        """Return the step's name."""
        return self._strategy.name

    def set_seed(self, seed: int) -> None:
        self._strategy.set_seed(seed)

    def bind_objective(self, objective: "DiversityObjective") -> None:
        """Hand the solve's primary diversity objective to this step's strategy; see `StrategyBase.bind_objective`."""
        self._strategy.bind_objective(objective)

    @abstractmethod
    def set_e2e_budget(self, e2e_budget: E2eBudget | None) -> None:
        """Take note of the solve's running end-to-end budget; the solver calls this on every step at solve start."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        state: SolverState,
        step_identity: SolverStepIdentity,
        progress_reporter: ProgressReporter | None = None,
        coordinator: "WorkerCoordinator | None" = None,
        batch_seconds: float = REPORTING_BATCH_SECONDS,
        elapsed_before_step: Elapsed = Elapsed(t_elapsed_sec=0.0, n_iterations=0),  # noqa: B008 -- immutable value
        intermediate_selections_enabled: bool = False,
    ) -> SolverStepResult:
        """Execute the solver step by running a strategy once or repeatedly, and return its result.

        Args:
            state: the mutable solver state the step reads and updates.
            step_identity: which step of the solve this run is; every checkpoint the step records
                carries it.
            progress_reporter: receives progress updates during the run; `None` disables reporting.
            coordinator: a `WorkerCoordinator` this step calls at each batch boundary; a step that
                runs as a single batch ignores it.
            batch_seconds: targeted wall-clock size of one batch.  Like `coordinator`, it is
                ignored by steps that run as a single batch.
            elapsed_before_step: what the solve had spent before this step started, so the step
                can tell the coordinator where it is on the solve-wide axis; its own checkpoints
                count from the step's start, and the solver shifts them afterwards.
            intermediate_selections_enabled: whether every checkpoint also carries a copy of the
                selection held at that moment.
        """
        raise NotImplementedError

    def get_debug_info(self) -> str:
        return self._strategy.get_debug_info()


# =================================================================================================
#  InitializationStep
# =================================================================================================
class InitializationStep(SolverStep[InitializationStrategy]):
    def __init__(self, init_strategy: InitializationStrategy) -> None:
        if not isinstance(init_strategy, InitializationStrategy):
            raise TypeError(
                "The provided strategy is not an InitializationStrategy. "
                + "Use one of the InitializationStrategy factory methods to instantiate one..",
            )
        super().__init__(init_strategy)

    def set_e2e_budget(self, e2e_budget: E2eBudget | None) -> None:
        """Ignore the end-to-end budget: initialization always runs to completion."""

    def run(
        self,
        state: SolverState,
        step_identity: SolverStepIdentity,
        progress_reporter: ProgressReporter | None = None,
        coordinator: "WorkerCoordinator | None" = None,
        batch_seconds: float = REPORTING_BATCH_SECONDS,
        elapsed_before_step: Elapsed = Elapsed(t_elapsed_sec=0.0, n_iterations=0),  # noqa: B008 -- immutable value
        intermediate_selections_enabled: bool = False,
    ) -> SolverStepResult:
        # --- set up progress tracking -----------
        progress_reporter = progress_reporter or SilentProgressReporter()
        duration = TargetDuration.iterations(int(state.k))  # we need to select k items
        tracker = duration.track()

        # --- execute initialization -------------
        with Timer() as t:
            while state.n_selected < state.k:
                # continue while we don't have a complete initial selection

                # --- update progress ------------
                progress_reporter.update(tracker.get_progress(), state, self.get_debug_info)

                # --- get next samples -----------
                samples = self._strategy.get_next_samples(
                    state=state,
                    k_remaining=state.k - state.n_selected,
                )

                # --- add items to state ---------
                state.add_many(samples, parallel=self._strategy.parallel_batch_add)

                tracker.report_iterations_done(len(samples))

        progress_reporter.solver_step_finished(tracker.get_progress(), state, self.get_debug_info)

        # --- gather results ---------------------
        return SolverStepResult(
            score_checkpoints=[
                ScoreCheckpoint.new(
                    step_identity,
                    Elapsed(t_elapsed_sec=t.t_elapsed_sec(), n_iterations=1),
                    state,
                    coordinator,
                    includes_selection=intermediate_selections_enabled,
                )
            ],
        )


# =================================================================================================
#  OptimizationStep
# =================================================================================================
class OptimizationStep(SolverStep[OptimizationStrategy]):
    def __init__(self, optim_strategy: OptimizationStrategy, duration: TargetDuration) -> None:
        if not isinstance(optim_strategy, OptimizationStrategy):
            raise TypeError(
                "The provided strategy is not an OptimizationStrategy. "
                + "Use one of the OptimizationStrategy factory methods to instantiate one..",
            )
        super().__init__(optim_strategy)
        self._duration = duration
        self._e2e_budget: E2eBudget | None = None

    def set_e2e_budget(self, e2e_budget: E2eBudget | None) -> None:
        """Store the solve's running e2e budget, replacing this step's duration with the e2e budget's remaining time."""
        self._e2e_budget = e2e_budget

    def _effective_duration(self) -> TargetDuration | None:
        """Return the duration this run gets — the e2e budget's remaining time where one is set.

        Return None, after raising a `SolverBudgetWarning`, when the e2e budget is already spent.

        NOTE: with several optimization steps under one e2e budget, each receives everything that
        remains, so an earlier step can starve the later ones.  Every preset produces exactly
        one optimization step; how several should share the remainder is undecided.
        """
        if self._e2e_budget is None:
            return self._duration
        remaining_sec = self._e2e_budget.remaining_sec()
        if remaining_sec <= 0.0:
            warnings.warn(
                f"The end-to-end budget of {self._e2e_budget.budget_sec}s was spent before optimization "
                "started; the returned selection is the initialization's.",
                SolverBudgetWarning,
                stacklevel=4,  # 4 frames up from here is the caller of solve(), whose budget it is
            )
            return None
        return TargetDuration.seconds(remaining_sec)

    def run(
        self,
        state: SolverState,
        step_identity: SolverStepIdentity,
        progress_reporter: ProgressReporter | None = None,
        coordinator: "WorkerCoordinator | None" = None,
        batch_seconds: float = REPORTING_BATCH_SECONDS,
        elapsed_before_step: Elapsed = Elapsed(t_elapsed_sec=0.0, n_iterations=0),  # noqa: B008 -- immutable value
        intermediate_selections_enabled: bool = False,
    ) -> SolverStepResult:
        """Iteratively improve the selection until the step's effective duration is spent.

        A step whose effective duration is already gone (see `_effective_duration`) is skipped,
        leaving the selection the earlier steps built.
        """
        # --- init -------------------------------
        progress_reporter = progress_reporter or SilentProgressReporter()
        duration = self._effective_duration()
        if duration is None:
            progress_reporter.solver_step_finished(None, state)
            return SolverStepResult(
                score_checkpoints=[
                    ScoreCheckpoint.new(
                        step_identity,
                        Elapsed(t_elapsed_sec=0.0, n_iterations=0),
                        state,
                        coordinator,
                        includes_selection=intermediate_selections_enabled,
                    )
                ]
            )
        tracker = duration.track()
        score_checkpoints: list[ScoreCheckpoint] = []
        next_checkpoint_iter_count = 1

        # --- main loop --------------------------
        while not (progress := tracker.get_progress()).is_finished:
            # --- update progress ----------------
            progress_reporter.update(
                progress,
                state,
                self.get_debug_info,
                ignore_infeasible_diversity=self._strategy.ignore_infeasible_diversity,
            )

            # --- do n iterations ----------------
            n_iters = self._determine_n_iterations(progress, next_checkpoint_iter_count, batch_seconds)
            self._strategy.perform_n_iterations(
                state=state,
                n_iters=n_iters,
                current_progress_frac=progress.fraction,
                progress_frac_per_iter=progress.est_progress_fraction_per_iter,
            )

            # --- report progress to tracker -----
            tracker.report_iterations_done(n_iters)

            # --- batch boundary -----------------
            if coordinator is not None:
                coordinator.at_batch_boundary(state, progress.fraction, elapsed_before_step + tracker.elapsed())

            # --- create checkpoint if needed ----
            if tracker.iter_count() >= next_checkpoint_iter_count:
                score_checkpoints.append(
                    ScoreCheckpoint.new(
                        step_identity,
                        tracker.elapsed(),
                        state,
                        coordinator,
                        includes_selection=intermediate_selections_enabled,
                    )
                )
                next_checkpoint_iter_count = int(
                    max(
                        [
                            next_checkpoint_iter_count + 1,
                            round(next_checkpoint_iter_count * 1.1),  # make checkpoint at every ~10% increment
                        ]
                    )
                )

        progress_reporter.solver_step_finished(
            progress,
            state,
            self.get_debug_info,
            ignore_infeasible_diversity=self._strategy.ignore_infeasible_diversity,
        )

        # --- gather results ---------------------
        elapsed = tracker.elapsed()
        if (len(score_checkpoints) == 0) or (elapsed.n_iterations > score_checkpoints[-1].elapsed.n_iterations):
            # make sure we always have a checkpoint after the last iteration
            score_checkpoints.append(
                ScoreCheckpoint.new(
                    step_identity,
                    elapsed,
                    state,
                    coordinator,
                    includes_selection=intermediate_selections_enabled,
                )
            )
        return SolverStepResult(score_checkpoints=score_checkpoints)

    @staticmethod
    def _determine_n_iterations(progress: Progress, next_checkpoint_iter_count: int, batch_seconds: float) -> int:
        """Determine number of iterations to execute in the next inner loop.

        We take into account:
          - estimated total number of iterations left in tracked duration
          - batch_seconds: the targeted wall-clock size of one batch
          - next_checkpoint_iter_count: this is the # of iterations at which we want to keep track
                                                                                  of the score we're optimizing.
        """
        iters_until_next_progress_report = int(batch_seconds * progress.est_iters_per_second)
        iters_until_next_checkpoint = next_checkpoint_iter_count - progress.iter_count
        half_iters_until_finished = progress.est_n_iters_remaining // 2  # iters to move 50% closer to being finished

        return max(
            1,  # never less than 1 iteration
            min(
                [
                    iters_until_next_progress_report,
                    iters_until_next_checkpoint,
                    half_iters_until_finished,
                ]
            ),
        )
