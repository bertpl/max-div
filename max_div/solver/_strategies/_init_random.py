from max_div.solver._solver_state import SolverState

from .core import SolverStrategy, StrategyType, iterations


class InitRandom(SolverStrategy):
    def __init__(self):
        super().__init__(
            strategy_type=StrategyType.INITIALIZATION,
            duration=iterations(1),
        )

    def do_one_iteration(self, state: SolverState):
        pass  # TODO
