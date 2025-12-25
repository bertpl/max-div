from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from max_div.internal.utils import ALMOST_ONE_F32
from max_div.sampling.poisson import sample_truncated_poisson
from max_div.solver._scheduling import ParameterSchedule, _evaluate_schedules, _schedules_to_2d_numpy_array
from max_div.solver._solver_state import SolverState
from max_div.solver._strategies._base import StrategyBase


# =================================================================================================
#  OptimizationStrategy
# =================================================================================================
class OptimizationStrategy(StrategyBase, ABC):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str | None = None, scheduled_params: dict[str, Any] | None = None):
        """
        Initialize the optimization strategy.
        :param name: optional name of the strategy; if omitted class name is used.
        :param scheduled_params: optional dictionary of parameters that have scheduled values.

                        The values of this dictionary are screened to check if any are of type ParameterSchedule.
                        These properties will be updated (using setattr(self, arg, value)) at the start of each
                        iteration, based on the ParameterSchedule and the current progress fraction.  All other
                        members of the dictionary are ignored and assumed not be scheduled; no action is taken f
                        or these.
        """
        super().__init__(name)
        if isinstance(scheduled_params, dict) and any(
            isinstance(v, ParameterSchedule) for v in scheduled_params.values()
        ):
            # at least 1 scheduled parameter
            self.has_scheduled_params = True
            self.scheduled_param_names = [k for k, v in scheduled_params.items() if isinstance(v, ParameterSchedule)]
            self.scheduled_param_configs = _schedules_to_2d_numpy_array(
                [v for v in scheduled_params.values() if isinstance(v, ParameterSchedule)]
            )
        else:
            # no scheduled parameters
            self.has_scheduled_params = False
            self.scheduled_params = []
            self.scheduled_param_configs = _schedules_to_2d_numpy_array([])

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def perform_n_iterations(
        self, state: SolverState, n_iters: int, current_progress_frac: float, progress_frac_per_iter: float
    ):
        """
        Perform n iterations of the optimization strategy, modifying the solver state in-place.
        :param state: (SolverState) current solver state to be modified and used to extract properties of current state.
        :param n_iters: (int) number of iterations to perform.
        :param current_progress_frac: (float) fraction in [0.0, 1.0] indicating current overall progress through total
                                      duration (iterations or time) configured for this SolverStep.
        :param progress_frac_per_iter: (float) fraction in [0.0, 1.0] indicating how much progress each iteration
                                       contributes towards the total duration configured for this SolverStep.  For time-based
                                       solver step configurations, this can be an estimate.
        """

        # --- prep ----------------------------------------
        if n_iters > 1:
            progress_frac_per_iter = min(
                progress_frac_per_iter,
                float(ALMOST_ONE_F32 * (1.0 - current_progress_frac) / (n_iters - 1)),
            )  # ensure we never progress beyond 1.0
        has_scheduled_params = self.has_scheduled_params

        # --- main loop -----------------------------------
        for _ in range(n_iters):
            # --- update scheduled parameters ---
            if has_scheduled_params:
                param_values = _evaluate_schedules(self.scheduled_param_configs, current_progress_frac)
                for param_name, param_value in zip(self.scheduled_param_names, param_values):
                    setattr(self, param_name, param_value)

            # --- execute iteration ---
            self._perform_single_iteration(state, current_progress_frac)

            # --- update progress ---
            current_progress_frac += progress_frac_per_iter

    @abstractmethod
    def _perform_single_iteration(self, state: SolverState, progress_frac: float):
        """
        Perform one iteration of the strategy, modifying the solver state in-place,
          trying to reach a more optimal solution.
        :param state: (SolverState) The current solver state.
        :param progress_frac: (float) Fraction in [0.0, 1.0] indicating current overall progress through total
                                 duration (iterations or time) configured for this SolverStep.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def random_swaps(cls) -> Self:
        from ._optim_random_swaps import OptimRandomSwaps

        return OptimRandomSwaps()


# =================================================================================================
#  Swap-Based Optimization Strategy base class
# =================================================================================================
class SwapBasedOptimizationStrategy(OptimizationStrategy, ABC):
    """
    Base class for swap-based optimization strategies, where in each iteration 'n' items are removed from the current
    selection and replaced by 'n' new items, but only if the swap improves the overall score.

    n is sampled from a truncated Poisson distribution with range [min_swap_size, max_swap_size] and lambda
    parameter swap_size_lambda, the latter of which can be set to a ParameterSchedule or a fixed value.

    Optionally constraint scores can be treated as soft constraints, in which case diversity score is mixed in with
    the constraint score to a certain degree.  This parameter can also be scheduled.

    Child classes need to implement the way in which...
      - n items are selected for removal from the current selection
      - n new items are selected for addition to the current selection
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        name: str | None = None,
        min_swap_size: int = 1,
        max_swap_size: int = 1,
        swap_size_lambda: float | ParameterSchedule = 1.0,
        constraint_softness: float | ParameterSchedule = 0.0,
        scheduled_params: dict[str, Any] | None = None,
    ):
        super().__init__(
            name=name,
            scheduled_params=(scheduled_params or dict())
            | dict(
                swap_size_lambda=swap_size_lambda,
                constraint_softness=constraint_softness,
            ),
        )
        self.min_swap_size: np.int32 = np.int32(min_swap_size)
        self.max_swap_size: np.int32 = np.int32(max_swap_size)
        self.swap_size_lambda = swap_size_lambda
        self.constraint_softness = constraint_softness

    # -------------------------------------------------------------------------
    #  Single Iteration
    # -------------------------------------------------------------------------
    def _perform_single_iteration(self, state: SolverState, progress_frac: float):
        """
        Perform one iteration of the swap-based optimization strategy:
          - determine swap size n
          - remove n samples from current selection
          - add n new samples to current selection
          - if this swap did not improve the score -> revert to previous selection
        """

        # --- prep ---
        seed = self.next_seed()  # we can use the same seed for all sampling in this iteration

        # --- determine swap size ---
        n = sample_truncated_poisson(
            self.min_swap_size,
            self.max_swap_size,
            np.float32(self.swap_size_lambda),
            seed=seed,
        )

        # --- take snapshot & remove n samples ---

        # snapshot
        state.set_snapshot()
        score_before = state.score

        # add
        samples_to_remove = self._samples_to_be_removed(state, n, seed)
        for s in samples_to_remove:
            state.remove(s)

        # --- add n samples and evaluate ---

        # add
        samples_to_add = self._samples_to_be_added(state, n, samples_just_removed=samples_to_remove, seed=seed)
        for s in samples_to_add:
            state.add(s)

        # evaluate
        score_after = state.score
        if score_after.as_tuple(self.constraint_softness) <= score_before.as_tuple(self.constraint_softness):
            state.restore_snapshot()  # score didn't improve -> restore snapshot

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def _samples_to_be_removed(
        self,
        state: SolverState,
        n: np.int32,
        seed: np.int64,
    ) -> NDArray[np.int32]:
        """
        Determine which n samples to remove from the current selection.  The values returned should be indices present
        in state.selected_index_array.
        :param state: (SolverState) current solver state, with # selected samples = k
        :param n: (np.int32) number of samples to be removed  (swap size)
        :param seed: (np.int64) random seed to be used for any sampling
        :return: (int32 ndarray) of shape (n,) with indices of samples to be REMOVED
        """
        raise NotImplementedError()

    @abstractmethod
    def _samples_to_be_added(
        self,
        state: SolverState,
        n: np.int32,
        samples_just_removed: NDArray[np.int32],
        seed: np.int64,
    ) -> NDArray[np.int32]:
        """
        Determine which n samples to add to the current selection, right after having removed n samples.
        The values returned should be indices present in state.not_selected_index_array.
        :param state: (SolverState) current solver state, with # selected samples = k-n
        :param n: (np.int32) number of samples to be added  (swap size)
        :param samples_just_removed: (int32 ndarray) of shape (n,) with indices of samples that were just removed
                                          (potentially to be taken into account to avoid resampling them)
        :param seed: (np.int64) random seed to be used for any sampling
        :return: (int32 ndarray) of shape (n,) with indices of samples to be ADDED
        """
        raise NotImplementedError()
