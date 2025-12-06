from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from max_div.solver._solver_state import SolverState


# =================================================================================================
#  OptimizationStrategy
# =================================================================================================
class OptimizationStrategy(ABC):
    # -------------------------------------------------------------------------
    #  Construction & Configuration
    # -------------------------------------------------------------------------
    def __init__(self, name: str | None = None):
        """
        Initialize the optimization strategy.
        :param name: optional name of the strategy
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def perform_n_iterations(self, state: SolverState, n: int):
        for _ in range(n):
            self._perform_single_iteration(state)

    @abstractmethod
    def _perform_single_iteration(self, state: SolverState):
        """
        Perform one iteration of the strategy, modifying the solver state in-place,
          trying to reach a more optimal solution.
        :param state: (SolverState) The current solver state.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def dummy(cls) -> Self:
        from ._optim_dummy import OptimDummy

        return OptimDummy()
