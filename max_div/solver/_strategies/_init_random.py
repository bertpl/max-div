from ._base import SolverStrategy, StrategyType


class InitRandom(SolverStrategy):
    def __init__(self):
        super().__init__(strategy_type=StrategyType.INITIALIZATION)
