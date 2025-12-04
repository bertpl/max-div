from unittest.mock import Mock

import pytest

from max_div.solver._strategies._base import (
    SolverStrategy,
    StrategyDuration,
    StrategyType,
    hours,
    iterations,
    minutes,
    seconds,
)


def test_strategy_duration_factory_methods():
    assert seconds(10)._max_seconds == 10
    assert seconds(10)._max_iterations is None

    assert minutes(2)._max_seconds == 120
    assert minutes(2)._max_iterations is None

    assert hours(1)._max_seconds == 3600
    assert hours(1)._max_iterations is None

    assert iterations(5)._max_iterations == 5
    assert iterations(5)._max_seconds is None


@pytest.mark.parametrize(
    "duration, n_iterations, n_seconds, expected",
    [
        (iterations(1000), 500, 100, False),
        (iterations(1000), 1000, 100, True),
        (seconds(10), 100, 5, False),
        (seconds(10), 100, 10, True),
        (minutes(1), 100, 30, False),
        (minutes(1), 100, 60, True),
        (hours(1), 100, 3500, False),
        (hours(1), 100, 3600, True),
        (StrategyDuration(None, None), 100, 100, False),
    ],
)
def test_strategy_duration_finished(duration: StrategyDuration, n_iterations: int, n_seconds: float, expected: bool):
    assert duration.finished(n_iterations, n_seconds) == expected


@pytest.mark.parametrize(
    "duration, n_iterations, n_seconds, expected",
    [
        (iterations(1000), 500, 100, (500, 1000)),
        (iterations(1000), 1000, 100, (1000, 1000)),
        (seconds(10), 100, 5, (5, 10)),
        (seconds(10), 100, 10, (10, 10)),
        (minutes(1), 100, 30, (30, 60)),
        (minutes(1), 100, 60, (60, 60)),
        (hours(1), 100, 3500, (3500, 3600)),
        (hours(1), 100, 3600, (3600, 3600)),
        (StrategyDuration(None, None), 100, 100, (0, 1)),
    ],
)
def test_strategy_duration_progress(
    duration: StrategyDuration, n_iterations: int, n_seconds: float, expected: tuple[int, int]
):
    assert duration.progress(n_iterations, n_seconds) == expected


def test_strategy_run():
    # --- arrange -----------------------------------------
    class TestStrategy(SolverStrategy):
        def __init__(self):
            super().__init__(
                strategy_type=StrategyType.INITIALIZATION,
                duration=iterations(1),
            )

        def do_one_iteration(self, state):
            state.add(1)

    strategy = TestStrategy()
    mock_state = Mock()

    # --- act ---------------------------------------------
    strategy.run(state=mock_state)

    # --- assert ------------------------------------------
    mock_state.add.assert_called_once_with(1)
