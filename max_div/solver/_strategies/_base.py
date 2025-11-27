from __future__ import annotations

from abc import ABC
from enum import StrEnum


class StrategyType(StrEnum):
    INITIALIZATION = "INITIALIZATION"
    OPTIMIZATION = "OPTIMIZATION"


class SolverStrategy(ABC):
    # -------------------------------------------------------------------------
    #  Construction & Configuration
    # -------------------------------------------------------------------------
    def __init__(self, strategy_type: StrategyType):
        """
        Initialize the solver strategy.
        :param strategy_type: type of the strategy
                               if StrategyType.INITIALIZATION, the class implements an initialization strategy.
                               if StrategyType.OPTIMIZATION,   the class implements an optimization strategy.
        """
        self._strategy_type = strategy_type

    @property
    def type(self) -> StrategyType:
        return self._strategy_type

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    pass  # TODO

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def init_random(cls) -> SolverStrategy:
        from ._init_random import InitRandom

        return InitRandom()

    @classmethod
    def optim_dummy(cls) -> SolverStrategy:
        from ._optim_dummy import OptimDummy

        return OptimDummy()
