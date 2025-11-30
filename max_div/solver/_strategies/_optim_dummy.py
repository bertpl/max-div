from ._base import SolverStrategy, StrategyType


class OptimDummy(SolverStrategy):
    def __init__(self):
        super().__init__(strategy_type=StrategyType.OPTIMIZATION)
