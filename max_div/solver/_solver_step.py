from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from max_div.internal.benchmarking._timer import Timer
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy

from ._duration import Elapsed, Progress, TargetDuration
from ._progress_reporting import ProgressReporter, SilentProgressReporter
from ._score import Score
from ._solver_state import SolverState
from ._strategies._base import StrategyBase


# =================================================================================================
#  SolverStepResult
# =================================================================================================
@dataclass
class SolverStepResult:
    # checkpoints of how score evolved during execution of the step
    # NOTE: we should always make sure the last checkpoint represents the final state after all iterations
    score_checkpoints: list[tuple[Elapsed, Score]]

    @property
    def elapsed(self) -> Elapsed:
        return self.score_checkpoints[-1][0]


# =================================================================================================
#  SolverStep
# =================================================================================================
S = TypeVar("S", bound=StrategyBase)


class SolverStep(ABC, Generic[S]):
    def __init__(self, strategy: S):
        self._strategy: S = strategy

    def name(self) -> str:
        return self._strategy.name

    def set_seed(self, seed: int):
        self._strategy.set_seed(seed)

    @abstractmethod
    def run(self, state: SolverState, progress_reporter: ProgressReporter | None = None) -> SolverStepResult:
        """Executes the solver step by executing a strategy 1x or repeatedly and returns a SolverStepResult."""
        raise NotImplementedError

    def get_debug_info(self) -> str:
        return self._strategy.get_debug_info()


# =================================================================================================
#  InitializationStep
# =================================================================================================
class InitializationStep(SolverStep[InitializationStrategy]):
    def __init__(self, init_strategy: InitializationStrategy):
        if not isinstance(init_strategy, InitializationStrategy):
            raise TypeError(
                "The provided strategy is not an InitializationStrategy. "
                + "Use one of the InitializationStrategy factory methods to instantiate one..",
            )
        super().__init__(init_strategy)

    def run(self, state: SolverState, progress_reporter: ProgressReporter | None = None) -> SolverStepResult:
        # --- set up progress tracking --------------------
        progress_reporter = progress_reporter or SilentProgressReporter()
        duration = TargetDuration.iterations(int(state.k))  # we need to select k items
        tracker = duration.track()

        # --- execute initialization ----------------------
        with Timer() as t:
            while state.n_selected < state.k:
                # continue while we don't have a complete initial selection

                # --- update progress ---
                progress_reporter.update(tracker.get_progress(), state, self.get_debug_info)

                # --- get next samples ---
                samples = self._strategy.get_next_samples(
                    state=state,
                    k_remaining=state.k - state.n_selected,
                )

                # --- add items to state ---
                state.add_many(samples)

                tracker.report_iterations_done(len(samples))

        progress_reporter.solver_step_finished(tracker.get_progress(), state, self.get_debug_info)

        # --- gather results ------------------------------
        return SolverStepResult(
            score_checkpoints=[
                (
                    Elapsed(
                        t_elapsed_sec=t.t_elapsed_sec(),
                        n_iterations=1,
                    ),
                    state.score,
                )
            ],
        )


# =================================================================================================
#  OptimizationStep
# =================================================================================================
class OptimizationStep(SolverStep[OptimizationStrategy]):
    def __init__(self, optim_strategy: OptimizationStrategy, duration: TargetDuration):
        if not isinstance(optim_strategy, OptimizationStrategy):
            raise TypeError(
                "The provided strategy is not an OptimizationStrategy. "
                + "Use one of the OptimizationStrategy factory methods to instantiate one..",
            )
        super().__init__(optim_strategy)
        self._duration = duration

    def run(self, state: SolverState, progress_reporter: ProgressReporter | None = None) -> SolverStepResult:
        # --- init ----------------------------------------
        progress_reporter = progress_reporter or SilentProgressReporter()
        tracker = self._duration.track()
        score_checkpoints: list[tuple[Elapsed, Score]] = []
        next_checkpoint_iter_count = 1

        # --- main loop -----------------------------------
        while not (progress := tracker.get_progress()).is_finished:
            # --- update progress ---
            progress_reporter.update(
                progress,
                state,
                self.get_debug_info,
                ignore_infeasible_diversity=self._strategy.ignore_infeasible_diversity,
            )

            # --- do n iterations ---
            n_iters = self._determine_n_iterations(progress, next_checkpoint_iter_count)
            self._strategy.perform_n_iterations(
                state=state,
                n_iters=n_iters,
                current_progress_frac=progress.fraction,
                progress_frac_per_iter=progress.est_progress_fraction_per_iter,
            )

            # --- report progress to tracker ---
            tracker.report_iterations_done(n_iters)

            # --- create checkpoint if needed ---
            if tracker.iter_count() >= next_checkpoint_iter_count:
                score_checkpoints.append((tracker.elapsed(), state.score))
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

        # --- gather results ------------------------------
        elapsed = tracker.elapsed()
        if (len(score_checkpoints) == 0) or (elapsed.n_iterations > score_checkpoints[-1][0].n_iterations):
            # make sure we always have a checkpoint after the last iteration
            score_checkpoints.append((elapsed, state.score))
        return SolverStepResult(score_checkpoints=score_checkpoints)

    @staticmethod
    def _determine_n_iterations(progress: Progress, next_checkpoint_iter_count: int) -> int:
        """
        Determine number of iterations to execute in the next inner loop.

        We take into account:
          - estimated total number of iterations left in tracked duration
          - we want to trigger a potential progress report at most every ~0.5sec
          - next_checkpoint_iter_count: this is the # of iterations at which we want to keep track
                                                                                  of the score we're optimizing.
        """
        iters_until_next_progress_report = int(0.5 * progress.est_iters_per_second)  # target = 2x/sec
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
