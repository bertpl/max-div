from unittest.mock import Mock

from max_div.solver._strategies import ProgressTracker, SolverStrategy, StrategyType, iterations


# =================================================================================================
#  Dummy Strategy
# =================================================================================================
class DummyStrategy(SolverStrategy):
    def __init__(self):
        super().__init__(
            strategy_type=StrategyType.INITIALIZATION,
            duration=iterations(1),
        )

    def do_one_iteration(self, state):
        state.add(1)


# =================================================================================================
#  Tests
# =================================================================================================
def test_strategy_properties():
    # --- arrange -----------------------------------------
    strategy = DummyStrategy()

    # --- act & assert ------------------------------------
    assert strategy.type == StrategyType.INITIALIZATION
    assert isinstance(strategy.duration, ProgressTracker)
    assert strategy.name == "DummyStrategy"


def test_strategy_run():
    # --- arrange -----------------------------------------
    strategy = DummyStrategy()
    mock_state = Mock()

    # --- act ---------------------------------------------
    strategy.run(state=mock_state, tqdm_desc="dummy_step")

    # --- assert ------------------------------------------
    mock_state.add.assert_called_once_with(1)
