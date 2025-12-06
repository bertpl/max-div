from abc import ABC, abstractmethod
from dataclasses import dataclass

from tqdm.auto import tqdm

from max_div.internal.benchmarking._timer import Timer
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy

from ._duration import Elapsed, ProgressTracker, TargetDuration
from ._score import Score
from ._solver_state import SolverState


# =================================================================================================
#  SolverStepResult
# =================================================================================================
@dataclass
class SolverStepResult:
    duration: Elapsed


# =================================================================================================
#  SolverStep
# =================================================================================================
class SolverStep(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

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
class InitializationStep(SolverStep):
    def __init__(self, init_strategy: InitializationStrategy):
        if not isinstance(init_strategy, InitializationStrategy):
            raise TypeError(
                "The provided strategy is not an InitializationStrategy. "
                + "Use one of the InitializationStrategy factory methods to instantiate one..",
            )
        self._strategy = init_strategy

    def name(self) -> str:
        return self._strategy.name

    def _run_child(self, state: SolverState, pbar: tqdm | None) -> SolverStepResult:
        with Timer() as t:
            self._strategy.initialize(state)

        return SolverStepResult(
            duration=Elapsed(
                t_elapsed_sec=t.t_elapsed_sec(),
                n_iterations=1,
            ),
        )


# =================================================================================================
#  OptimizationStep
# =================================================================================================
class OptimizationStep(SolverStep):
    def __init__(self, optim_strategy: OptimizationStrategy, duration: TargetDuration):
        if not isinstance(optim_strategy, OptimizationStrategy):
            raise TypeError(
                "The provided strategy is not an OptimizationStrategy. "
                + "Use one of the OptimizationStrategy factory methods to instantiate one..",
            )
        self._strategy = optim_strategy
        self._duration = duration

    def name(self) -> str:
        return self._strategy.name

    def _run_child(self, state: SolverState, pbar: tqdm | None) -> SolverStepResult:
        # --- init ----------------------------------------
        tracker = self._duration.track()
        checkpoints: list[tuple[Elapsed, Score]] = []
        next_checkpoint_iter_count = 1

        # --- main loop -----------------------------------
        while not (progress := tracker.get_progress()).is_finished:
            # --- update progress ---
            if pbar:
                progress.update_tqdm(pbar)

            # --- do n iterations ---
            n_iters = self._determine_n_iterations(tracker, next_checkpoint_iter_count)
            self._strategy.perform_n_iterations(state, n_iters)

            # --- create checkpoint if needed ---
            if tracker.iter_count() >= next_checkpoint_iter_count:
                checkpoints.append((tracker.elapsed(), state.score))
                next_checkpoint_iter_count = int(
                    max(
                        [
                            next_checkpoint_iter_count + 1,
                            round(next_checkpoint_iter_count * 1.1),  # make checkpoint at every ~10% increment
                        ]
                    )
                )

            # --- update progress ---
            tracker.iterations_done(n_iters)

        # --- finalize ------------------------------------
        if pbar:
            progress.update_tqdm(pbar)  # one last time
        return SolverStepResult(
            duration=tracker.elapsed(),
        )

    @staticmethod
    def _determine_n_iterations(tracker: ProgressTracker, next_checkpoint_iter_count: int) -> int:
        """
        Determine number of iterations to execute in the next inner loop.

        We take into account:
          - estimated total number of iterations left in tracked duration
          - we want to show a progress bar update every ~1sec
          - next_checkpoint_iter_count: this is the # of iterations at which we want to keep track
                                                                                  of the score we're optimizing.
        """
        total_iters_left = tracker.estimated_n_iterations_remaining()
        iters_per_second = tracker.iters_per_second()
        iter_count = tracker.iter_count()

        return max(
            1,  # never less than 1 iteration
            min(
                [
                    int(iters_per_second),  # so we can report progress every 1sec
                    next_checkpoint_iter_count - iter_count,  # so we can make a checkpoint at exactly the right time
                    int(total_iters_left / 2),  # proceed towards the end in steps of 50% of what's remaining at most
                ]
            ),
        )
