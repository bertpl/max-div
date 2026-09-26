from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from numpy._typing import NDArray

from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._strategies import InitializationStrategy

_FACTORY_IDS = ["farthest_point", "fixed_selection", "most_feasible", "random_selection", "random_selection(uncon)"]


@pytest.mark.parametrize(
    "factory_method",
    [
        partial(InitializationStrategy.farthest_point, top_k=4),
        partial(InitializationStrategy.fixed_selection, [0, 1, 2]),
        partial(InitializationStrategy.most_feasible),
        partial(InitializationStrategy.random_selection, ignore_constraints=False),
        partial(InitializationStrategy.random_selection, ignore_constraints=True),
    ],
    ids=_FACTORY_IDS,
)
def test_initialization_strategy_factory(factory_method: Callable[[], InitializationStrategy]):
    """Test factory methods of InitializationStrategy base class."""

    # --- act & assert -----------------
    assert isinstance(factory_method(), InitializationStrategy)


def test_initialization_strategy_factory_ids_name_every_factory_method():
    """The factory test's ids name every factory classmethod of `InitializationStrategy`."""
    # --- arrange / act ----------------
    factory_names = {name for name, member in vars(InitializationStrategy).items() if isinstance(member, classmethod)}
    tested_names = {factory_id.split("(")[0] for factory_id in _FACTORY_IDS}

    # --- assert -----------------------
    assert tested_names == factory_names


def test_initialization_strategy_properties():
    # --- arrange ----------------------
    class TestInitializationStrategy(InitializationStrategy):
        def __init__(self, name: str | None = None):
            super().__init__(name)

        def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
            pass

    # --- act & assert -----------------
    assert TestInitializationStrategy().name == "TestInitializationStrategy"
    assert TestInitializationStrategy("custom_name").name == "custom_name"
