from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from max_div.internal.utils import ALMOST_ONE_F32
from max_div.solver._solver_state import SolverState
from max_div.solver._strategies._base import StrategyBase


# =================================================================================================
#  OptimizationStrategy
# =================================================================================================
class OptimizationStrategy(StrategyBase, ABC):
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

        # --- prep ---
        progress_frac_per_iter = min(
            progress_frac_per_iter,
            float(ALMOST_ONE_F32 * (1.0 - current_progress_frac) / n_iters),
        )  # ensure we never progress beyond 1.0

        # --- main loop ---
        for _ in range(n_iters):
            self._perform_single_iteration(state, current_progress_frac)
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
