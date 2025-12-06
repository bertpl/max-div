from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from max_div.solver._solver_state import SolverState


# =================================================================================================
#  InitializationStrategy
# =================================================================================================
class InitializationStrategy(ABC):
    # -------------------------------------------------------------------------
    #  Construction & Configuration
    # -------------------------------------------------------------------------
    def __init__(self, name: str | None = None):
        """
        Initialize the initialization strategy.
        :param name: optional name of the strategy
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
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
    def random(cls) -> Self:
        from ._init_random import InitRandom

        return InitRandom()
