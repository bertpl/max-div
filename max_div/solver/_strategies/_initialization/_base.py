from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from max_div.solver._solver_state import SolverState
from max_div.solver._strategies._base import StrategyBase


# =================================================================================================
#  InitializationStrategy
# =================================================================================================
class InitializationStrategy(StrategyBase, ABC):
    @abstractmethod
    def initialize(self, state: SolverState):
        """
        Computes an initial solution, starting from a SolverState with empty selection,
          resulting in a SolverState with a selection of appropriate size.
        :param state: (SolverState) The current solver state.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def one_shot_random(cls, constrained: bool = True, uniform: bool = False) -> Self:
        """Create a InitOneShotRandom initialization strategy."""
        from ._init_one_shot_random import InitOneShotRandom

        return InitOneShotRandom(
            constrained=constrained,
            uniform=uniform,
        )
