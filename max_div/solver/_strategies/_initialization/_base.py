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
    def dummy(cls) -> Self:
        """Create a InitDummy initialization strategy."""
        from ._init_dummy import InitDummy

        return InitDummy()

    @classmethod
    def random_one_shot(cls, constrained: bool = True, uniform: bool = False) -> Self:
        """Create a InitRandomOneShot initialization strategy."""
        from ._init_random_one_shot import InitRandomOneShot

        return InitRandomOneShot(
            constrained=constrained,
            uniform=uniform,
        )

    @classmethod
    def random_batched(cls, b: int, constrained: bool = True) -> Self:
        """Create a InitRandomBatched initialization strategy."""
        from ._init_random_batched import InitRandomBatched

        return InitRandomBatched(
            b=b,
            constrained=constrained,
        )

    @classmethod
    def eager(cls, nc: int) -> Self:
        """Create a InitEager initialization strategy."""
        from ._init_eager import InitEager

        return InitEager(nc=nc)
