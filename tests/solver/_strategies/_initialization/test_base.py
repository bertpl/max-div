from max_div.solver._solver_state import SolverState
from max_div.solver._strategies import InitializationStrategy


def test_initialization_strategy_factory():
    assert isinstance(InitializationStrategy.one_shot_random(), InitializationStrategy)


def test_initialization_strategy_properties():
    # --- arrange -----------------------------------------
    class TestInitializationStrategy(InitializationStrategy):
        def __init__(self, name: str | None = None):
            super().__init__(name)

        def initialize(self, state: SolverState):
            pass

    # --- act & assert ------------------------------------
    assert TestInitializationStrategy().name == "TestInitializationStrategy"
    assert TestInitializationStrategy("custom_name").name == "custom_name"
