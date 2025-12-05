import time

import pytest

from max_div.solver._strategies import StrategyDuration, hours, iterations, minutes, seconds
from max_div.solver._strategies.core import DurationProgress


# =================================================================================================
#  IterationBased
# =================================================================================================
def test_iteration_based_duration_factory_methods():
    assert isinstance(iterations(5), StrategyDuration)
    assert iterations(5)._max_iters == 5


def test_iteration_based_duration_progress():
    # --- arrange -----------------------------------------
    duration_1 = iterations(1)
    duration_2 = iterations(2)
    duration_3 = iterations(1000)

    # --- act & assert 1 ----------------------------------
    duration_1.start()
    duration_2.start()
    duration_3.start()
    time.sleep(0.1)  # should not do anything, since these are iteration-based
    assert duration_1.progress() == DurationProgress(0, 1, False)
    assert duration_2.progress() == DurationProgress(0, 2, False)
    assert duration_3.progress() == DurationProgress(0, 1000, False)

    # --- act & assert 2 ----------------------------------
    duration_1.iteration_done()
    duration_2.iteration_done()
    duration_3.iteration_done()
    assert duration_1.progress() == DurationProgress(1, 1, True)
    assert duration_2.progress() == DurationProgress(1, 2, False)
    assert duration_3.progress() == DurationProgress(1, 1000, False)

    # --- act & assert 3 ----------------------------------
    duration_1.iteration_done()
    duration_2.iteration_done()
    duration_3.iteration_done()
    assert duration_1.progress() == DurationProgress(2, 1, True)
    assert duration_2.progress() == DurationProgress(2, 2, True)
    assert duration_3.progress() == DurationProgress(2, 1000, False)


# =================================================================================================
#  TimeBased
# =================================================================================================
def test_time_based_duration_factory_methods():
    assert isinstance(seconds(5), StrategyDuration)
    assert isinstance(minutes(1.5), StrategyDuration)
    assert isinstance(hours(1.25), StrategyDuration)

    assert seconds(5)._max_seconds == 5
    assert minutes(1.5)._max_seconds == 90
    assert hours(1.25)._max_seconds == 4500


def test_time_based_duration_progress():
    # --- arrange -----------------------------------------
    duration_1 = seconds(0.001)
    duration_2 = seconds(0.01)
    duration_3 = seconds(10)

    # --- act & assert 1 ----------------------------------
    duration_1.start()
    duration_2.start()
    duration_3.start()
    duration_1.iteration_done()  # should not do anything, since these are time-based
    duration_2.iteration_done()  # should not do anything, since these are time-based
    duration_3.iteration_done()  # should not do anything, since these are time-based
    assert duration_1.progress() == DurationProgress(0, 0, False)
    assert duration_2.progress() == DurationProgress(0, 0, False)
    assert duration_3.progress() == DurationProgress(0, 10, False)

    # --- act & assert 2 ----------------------------------
    time.sleep(0.002)
    assert duration_1.progress() == DurationProgress(0, 0, True)
    assert duration_2.progress() == DurationProgress(0, 0, False)
    assert duration_3.progress() == DurationProgress(0, 10, False)

    # --- act & assert 3 ----------------------------------
    time.sleep(0.02)
    assert duration_1.progress() == DurationProgress(0, 0, True)
    assert duration_2.progress() == DurationProgress(0, 0, True)
    assert duration_3.progress() == DurationProgress(0, 10, False)
