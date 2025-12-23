from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from max_div.internal.utils import ALMOST_ONE_F32
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
    def dummy(cls) -> Self:
        from ._optim_dummy import OptimDummy

        return OptimDummy()
