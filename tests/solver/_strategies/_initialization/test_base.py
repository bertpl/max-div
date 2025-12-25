from functools import partial
from typing import Callable

import pytest

from max_div.solver._solver_state import SolverState
from max_div.solver._strategies import InitializationStrategy


@pytest.mark.parametrize(
    "factory_method",
    [
        InitializationStrategy.dummy,
        InitializationStrategy.random_one_shot,
        partial(InitializationStrategy.random_batched, b=2),
        partial(InitializationStrategy.eager, nc=2),
    ],
    ids=["dummy", "random_one_shot", "random_batched", "eager"],
)
def test_initialization_strategy_factory(factory_method: Callable[[], InitializationStrategy]):
    """Test factory methods of InitializationStrategy base class."""

    # --- act & assert ------------------------------------
    assert isinstance(factory_method(), InitializationStrategy)


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
