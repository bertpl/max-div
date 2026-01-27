from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numba
import numpy as np
from numpy.typing import NDArray

from max_div.internal.utils import ALMOST_ONE_F32
from max_div.solver._parameters import (
    AdaptiveSampler,
    ParameterSchedule,
    ParameterValueSource,
    _evaluate_schedules,
    _schedules_to_2d_numpy_array,
)
from max_div.solver._solver_state import SolverState
from max_div.solver._strategies._base import StrategyBase

ParamValueType = ParameterValueSource | float | int | np.float32 | np.int32 | bool


# =================================================================================================
#  OptimizationStrategy
# =================================================================================================
class OptimizationStrategy(StrategyBase, ABC):
    # -------------------------------------------------------------------------
    #  Constructor / Configuration
    # -------------------------------------------------------------------------
    def __init__(
        self,
        name: str | None = None,
        dynamic_params: dict[str, ParamValueType] | None = None,
        ignore_infeasible_diversity_up_to_fraction: float = -1.0,
    ):
        """
        Initialize the optimization strategy.
        :param name: optional name of the strategy; if omitted class name is used.
        :param dynamic_params: optional dictionary of parameters that are potentially adjusted each iteration.

                The values of this dictionary are triaged by type, with different action taken:

                ParameterSchedule       these parameters are updated each iteration based on a schedule,
                                        which essentially maps progress_fraction -> parameter value.

                AdaptiveSampler         these parameters are sampled from a distribution in each iteration,
                                        which is adapted based on success/failure of the resulting swaps.

                other (float, int, ...) these parameters are fixed for the duration of the strategy.

        :param ignore_infeasible_diversity_up_to_fraction: float between 0 and 1.  If provided, we compare solution
                           scores using the flag 'ignore_infeasible_diversity' until the step has progressed up to this
                           fraction.  This allows optimization strategies to focus on satisfying constraints first
                           before trying to improve diversity.  Trying to improve diversity while still infeasible wrt
                           constraints, can steer away the solution from more feasible regions of the search space,
                           in case of hard-to-satisfy, overlapping constraints.

                           This parameter will influence how the instance field 'ignore_infeasible_diversity' is set.
                           When not provided it is always False; when provided it will start as True and will get set
                           to False at the appropriate time during optimization.

        """
        super().__init__(name)
        self.ignore_infeasible_diversity_up_to_fraction = ignore_infeasible_diversity_up_to_fraction
        self.ignore_infeasible_diversity = 0.0 <= ignore_infeasible_diversity_up_to_fraction

        # --- initialize scheduled parameters ---
        #  --> first initialize as if we don't have any; potentially overridden by _configure_dynamic_params
        self.has_scheduled_params = False
        self.scheduled_params = []
        self.scheduled_param_configs = np.empty(0, dtype=np.float32)

        # --- adaptively sampled parameters ---
        #  --> first initialize as if we don't have any; potentially overridden by _configure_dynamic_params
        self.has_sampled_params = False
        self.sampled_params: dict[str, AdaptiveSampler] = dict()

        # --- now actually configure them ---
        if dynamic_params:
            self._configure_dynamic_params(dynamic_params)

        # --- iteration counts ---
        self.iter = 0

    def _configure_dynamic_params(self, dynamic_params: dict[str, ParamValueType]):
        """
        Internal method to configure dynamic parameters (scheduled and/or sampled).
        :param dynamic_params: (dict) dictionary of dynamic parameters to configure
        """

        # --- schedule parameters ---------------
        scheduled_parameters = {
            param_name: param_value
            for param_name, param_value in dynamic_params.items()
            if isinstance(param_value, ParameterSchedule)
        }
        if scheduled_parameters:
            self.has_scheduled_params = True
            self.scheduled_param_names = list(scheduled_parameters.keys())
            self.scheduled_param_configs = _schedules_to_2d_numpy_array(list(scheduled_parameters.values()))

        # --- sampled parameters ----------------
        sampled_parameters = {
            param_name: param_value
            for param_name, param_value in dynamic_params.items()
            if isinstance(param_value, AdaptiveSampler)
        }
        if sampled_parameters:
            self.has_sampled_params = True
            self.sampled_params = sampled_parameters

    @property
    def has_dynamic_params(self) -> bool:
        return self.has_scheduled_params or self.has_sampled_params

    def set_seed(self, seed: int | np.int64) -> None:
        super().set_seed(seed)
        for i_sampler, sampler in enumerate(self.sampled_params.values()):
            # set seed for each sampler, offset by multiple of large prime
            sampler.update_seed(seed + (1_234_577 * i_sampler))

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
        has_sampled_params = self.has_sampled_params

        # --- main loop -----------------------------------
        for _ in range(n_iters):
            # --- update dynamic parameters ---
            if has_scheduled_params:
                param_values = _evaluate_schedules(self.scheduled_param_configs, current_progress_frac)
                for param_name, param_value in zip(self.scheduled_param_names, param_values):
                    setattr(self, param_name, param_value)
            if has_sampled_params:
                for param_name, sampler in self.sampled_params.items():
                    param_value = sampler.new_sample()
                    setattr(self, param_name, param_value)
            self.ignore_infeasible_diversity = current_progress_frac <= self.ignore_infeasible_diversity_up_to_fraction

            # --- execute iteration ---
            success = self._perform_single_iteration(state, current_progress_frac)

            # --- inform samplers of success/failure ---
            if has_sampled_params:
                for sampler in self.sampled_params.values():
                    sampler.feedback(success)

            # --- update progress ---
            self.iter += 1
            current_progress_frac += progress_frac_per_iter

    @abstractmethod
    def _perform_single_iteration(self, state: SolverState, progress_frac: float) -> bool:
        """
        Perform one iteration of the strategy, modifying the solver state in-place,
          trying to reach a more optimal solution.
        :param state: (SolverState) The current solver state.
        :param progress_frac: (float) Fraction in [0.0, 1.0] indicating current overall progress through total
                                 duration (iterations or time) configured for this SolverStep.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def initial_param_value(param: ParamValueType) -> float:
        """
        Helper method to get the initial value of a parameter that may be either dynamic or fixed.
        Intended for use inside constructors of child classes.
        :param param: (ParamValueType) parameter to get initial value for
        :return: (float) initial value of the parameter
        """
        if isinstance(param, ParameterValueSource):
            return param.get_initial_value()
        else:
            return float(param)

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def random_swaps(cls) -> Self:
        from ._optim_random_swaps import OptimRandomSwaps

        return OptimRandomSwaps()

    @classmethod
    def guided_swaps(
        cls,
        min_swap_size: int = 1,
        max_swap_size: int = 1,
        swap_size_lambda: float | ParameterSchedule = 1.0,
        constraint_softness: float | ParameterSchedule = 0.0,
        p_add_constraint_aware: float | ParameterSchedule = 1.0,
        remove_selectivity_modifier: float | ParameterSchedule = 0.0,
        add_selectivity_modifier: float | ParameterSchedule = 0.0,
    ) -> Self:
        from ._optim_guided_swaps import OptimGuidedSwaps

        return OptimGuidedSwaps(
            min_swap_size=min_swap_size,
            max_swap_size=max_swap_size,
            swap_size_lambda=swap_size_lambda,
            constraint_softness=constraint_softness,
            p_add_constraint_aware=p_add_constraint_aware,
            remove_selectivity_modifier=remove_selectivity_modifier,
            add_selectivity_modifier=add_selectivity_modifier,
        )

    @classmethod
    def smart_swaps(
        cls,
        swap_size_max: int,
        nc_remove_max: int,
        nc_add_max: int,
        tau_learn: float = 100.0,
        ignore_infeasible_diversity_up_to_fraction: float = -1.0,
        cost_awareness: float = 0.0,
    ) -> Self:
        from ._optim_smart_swaps import OptimSmartSwaps

        return OptimSmartSwaps(
            swap_size_max=swap_size_max,
            nc_remove_max=nc_remove_max,
            nc_add_max=nc_add_max,
            tau_learn=tau_learn,
            ignore_infeasible_diversity_up_to_fraction=ignore_infeasible_diversity_up_to_fraction,
            cost_awareness=cost_awareness,
        )


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
        constraint_softness: float | ParameterValueSource = 0.0,
        dynamic_params: dict[str, ParamValueType] | None = None,
        ignore_infeasible_diversity_up_to_fraction: float = -1.0,
    ):
        super().__init__(
            name=name,
            dynamic_params=(dynamic_params or dict())
            | dict(
                constraint_softness=constraint_softness,
            ),
            ignore_infeasible_diversity_up_to_fraction=ignore_infeasible_diversity_up_to_fraction,
        )
        self.constraint_softness: float = self.initial_param_value(constraint_softness)
        self._success_rate_state = np.zeros(20, dtype=np.int64)  # buffer for last 10 success & 10 fail iters

    def get_success_rate(self) -> float:
        """
        Estimate the swap success rate of the optimization strategy, based on recent history of successes and failures.
        :return: (float) estimated success rate in range [0.0, 1.0]
        """
        return float(_estimate_success_rate(self._success_rate_state))

    # -------------------------------------------------------------------------
    #  Single Iteration
    # -------------------------------------------------------------------------
    def _perform_single_iteration(self, state: SolverState, progress_frac: float) -> bool:
        """
        Perform one iteration of the swap-based optimization strategy:
          - determine swap size n
          - remove n samples from current selection
          - add n new samples to current selection
          - if this swap did not improve the score -> revert to previous selection
        """

        # --- init ---
        n_swap = min(
            self._determine_swap_size(),
            state.n_selected,  # we need to remove n_swap samples from selected, so n_selected is lower bound
            state.n - state.n_selected,  # we need to select n_swap from currently unselected, so this is lower bound
        )

        # do this now, before the just-removed ones get added to the not_selected ones; we don't want these
        # just-removed ones to be selected to be added again immediately.
        candidate_samples_to_add = state.not_selected_index_array

        # --- take snapshot & remove n samples ---

        # score before
        state.set_snapshot()
        score_before = state.score
        score_tuple_before = score_before.as_tuple(
            soft=self.constraint_softness,
            ignore_infeasible_diversity=self.ignore_infeasible_diversity,
        )

        # remove
        _ = self._remove_samples(state, n_swap)

        # add
        _ = self._add_samples(state, n_swap, candidate_samples_to_add)

        # evaluate
        score_after = state.score
        score_tuple_after = score_after.as_tuple(
            soft=self.constraint_softness,
            ignore_infeasible_diversity=self.ignore_infeasible_diversity,
        )
        if score_tuple_after < score_tuple_before:
            # score deteriorated -> revert; failure
            state.restore_snapshot()
            success = False
        else:
            # score did improve or stayed equal -> do not revert; success
            # NOTE: for the sake of exploring the search space, swaps with equal score are not reverted.
            success = True

        # --- update fail/success history ---
        _update_success_rate_state(self._success_rate_state, success)

        # --- return success flag ---
        return success

    # -------------------------------------------------------------------------
    #  Internal methods that can be overridden
    # -------------------------------------------------------------------------
    def _remove_samples(self, state: SolverState, n_to_remove: np.int32) -> NDArray[np.int32]:
        """
        REMOVE n samples and return the indices of removed samples.
        """
        samples_to_remove = self._samples_to_be_removed(state, n_to_remove)
        state.remove_many(samples_to_remove)
        return samples_to_remove

    def _add_samples(
        self, state: SolverState, n_to_add: np.int32, candidate_samples: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """
        ADD n samples and return the indices of added samples.
        """
        samples_to_add = self._samples_to_be_added(state, n_to_add, candidate_samples)
        state.add_many(samples_to_add)
        return samples_to_add

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def _determine_swap_size(self) -> np.int32:
        """
        Determine the swap size n for the current iteration.
        :return: (np.int32) swap size n
        """
        raise NotImplementedError()

    @abstractmethod
    def _samples_to_be_removed(self, state: SolverState, n_to_remove: np.int32) -> NDArray[np.int32]:
        """
        Determine which n samples to remove from the current selection.  The values returned should be indices present
        in state.selected_index_array.

        NOTE: for reproducibility, any random sampling inside this method should use self.next_seed() method of the
              strategy to get a new seed.

        :param state: (SolverState) current solver state, with # selected samples = k
        :param n_to_remove: (np.int32) number of samples to be removed  (swap size)
        :return: (int32 ndarray) of shape (n,) with indices of samples to be REMOVED
        """
        raise NotImplementedError()

    @abstractmethod
    def _samples_to_be_added(
        self, state: SolverState, n_to_add: np.int32, candidate_samples: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """
        Determine which n samples to add to the current selection, right after having removed n samples.
        The values returned should be indices present in state.not_selected_index_array.

        NOTE: for reproducibility, any random sampling inside this method should use self.next_seed() method of the
              strategy to get a new seed.

        :param state: (SolverState) current solver state, with # selected samples = k-n_to_add
        :param n_to_add: (np.int32) number of samples to be added  (swap size)
        :param candidate_samples: (1D np.int32 ndarray) with candidate samples to choose from; this method should
                                     NEVER return samples that are not in this array.
        :return: (int32 ndarray) of shape (n,) with indices of samples to be ADDED
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Debug info
    # -------------------------------------------------------------------------
    def get_debug_info(self) -> str:
        success_rate = self.get_success_rate()
        return f"scs={100 * success_rate:7.3f}%".ljust(100)


# =================================================================================================
#  Helper classes
# =================================================================================================
@numba.njit("void(int64[:], boolean)", fastmath=True, inline="always", cache=True)
def _update_success_rate_state(success_rate_state: NDArray[np.int64], success: bool):
    """
    Update success rate state in-place, based on provided success flag.  Current iteration # is estimated based on
    values found in the state (current iter = max(success_rate_state) + 1).
    :param success_rate_state: 2n-sized np.int64 array representing buffer of
                                 - n last success iters (in order)
                                 - n last fail iters (in order)
    :param success: (bool) True if latest iteration was a success, False otherwise
    """
    n = int(success_rate_state.shape[0] // 2)
    it = max(success_rate_state[n - 1], success_rate_state[-1]) + 1
    if success:
        # shift success history to the left & add new iter at the end
        success_rate_state[0 : n - 1] = success_rate_state[1:n]
        success_rate_state[n - 1] = it
    else:
        # shift failure history to the left & add new iter at the end
        success_rate_state[n : 2 * n - 1] = success_rate_state[n + 1 : 2 * n]
        success_rate_state[2 * n - 1] = it


@numba.njit("float64(int64[:])", fastmath=True, inline="always", cache=True)
def _estimate_success_rate(success_rate_state: NDArray[np.int64]) -> np.float64:
    """
    Estimates the success rate based on the provided success rate state.

    success_rate is computed as:

        success_rate_proxy = 1/(current_iter - mean(success_rate_state[:n]))
        failure_rate_proxy = 1/(current_iter - mean(success_rate_state[n:]))

        success_rate = success_rate_proxy / (success_rate_proxy + failure_rate_proxy)

    :param success_rate_state: 2n-sized np.int64 array representing buffer of
                                 - n last success iters (in order)
                                 - n last fail iters (in order)
    :return: (float64) estimated success rate in range [0.0, 1.0]
    """
    n = int(success_rate_state.shape[0] // 2)
    it = max(success_rate_state[n - 1], success_rate_state[-1]) + 1
    success_rate_proxy = 1.0 / (it - np.mean(success_rate_state[0:n]))
    failure_rate_proxy = 1.0 / (it - np.mean(success_rate_state[n : 2 * n]))
    success_rate = success_rate_proxy / (success_rate_proxy + failure_rate_proxy)
    return success_rate
