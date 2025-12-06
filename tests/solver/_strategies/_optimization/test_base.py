from max_div.solver._solver_state import SolverState
from max_div.solver._strategies import OptimizationStrategy


def test_optimization_strategy_factory():
    assert isinstance(OptimizationStrategy.dummy(), OptimizationStrategy)


def test_optimization_strategy_properties():
    # --- arrange -----------------------------------------
    class TestOptimizationStrategy(OptimizationStrategy):
        def __init__(self, name: str | None = None):
            super().__init__(name)

        def _perform_single_iteration(self, state: SolverState):
            pass

    # --- act & assert ------------------------------------
    assert TestOptimizationStrategy().name == "TestOptimizationStrategy"
    assert TestOptimizationStrategy("custom_name").name == "custom_name"
