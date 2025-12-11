from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from tqdm.auto import tqdm

from max_div.internal.benchmarking._timer import Timer
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy

from ._duration import Elapsed, Progress, TargetDuration
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
        self._strategy.seed = seed

    def run(self, state: SolverState, tqdm_desc: str | None = None) -> SolverStepResult:
        """
        Executes the solver step by executing a strategy 1x or repeatedly and returns a SolverStepResult.
        """

        # --- init ---
        pbar = tqdm(desc=tqdm_desc, total=1) if (tqdm_desc is None) else None

        # --- execute child ---
        result = self._run_child(state, pbar)

        # --- wrap up ---
        if (pbar is not None) and (pbar.n < pbar.total):
            pbar.n = pbar.total
            pbar.refresh()
        return result

    @abstractmethod
    def _run_child(self, state: SolverState, pbar: tqdm | None) -> SolverStepResult:
        raise NotImplementedError


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

    def _run_child(self, state: SolverState, pbar: tqdm | None) -> SolverStepResult:
        # --- execute initialization ----------------------
        with Timer() as t:
            self._strategy.initialize(state)

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

    def _run_child(self, state: SolverState, pbar: tqdm | None) -> SolverStepResult:
        # --- init ----------------------------------------
        tracker = self._duration.track()
        score_checkpoints: list[tuple[Elapsed, Score]] = []
        next_checkpoint_iter_count = 1

        # --- main loop -----------------------------------
        while not (progress := tracker.get_progress()).is_finished:
            # --- update progress ---
            if pbar:
                progress.update_tqdm(pbar)

            # --- do n iterations ---
            n_iters = self._determine_n_iterations(progress, next_checkpoint_iter_count)
            self._strategy.perform_n_iterations(
                state=state,
                n_iters=n_iters,
                current_progress_frac=progress.fraction,
                progress_frac_per_iter=progress.est_progress_fraction_per_iter,
            )

            # --- update progress ---
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

        if pbar:
            progress.update_tqdm(pbar)  # one last time

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
          - we want to show a progress bar update every ~1sec
          - next_checkpoint_iter_count: this is the # of iterations at which we want to keep track
                                                                                  of the score we're optimizing.
        """
        iters_until_next_pbar_update = int(progress.est_iters_per_second)  # target = 1x/sec
        iters_until_next_checkpoint = next_checkpoint_iter_count - progress.iter_count
        half_iters_until_finished = progress.est_n_iters_remaining // 2  # iters to move 50% closer to being finished

        return max(
            1,  # never less than 1 iteration
            min(
                [
                    iters_until_next_pbar_update,
                    iters_until_next_checkpoint,
                    half_iters_until_finished,
                ]
            ),
        )
