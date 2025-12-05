from .core import SolverStrategy, StrategyType, iterations


class OptimDummy(SolverStrategy):
    def __init__(self):
        super().__init__(
            strategy_type=StrategyType.OPTIMIZATION,
            duration=iterations(1),
        )

    def do_one_iteration(self, state):
        pass  # TODO
