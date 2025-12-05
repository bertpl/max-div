import time

import pytest

from max_div.solver._strategies import ProgressTracker, hours, iterations, minutes, seconds
from max_div.solver._strategies.core import Progress


# =================================================================================================
#  IterationBased
# =================================================================================================
def test_iteration_based_progress_factory_methods():
    assert isinstance(iterations(5), ProgressTracker)
    assert iterations(5)._max_iters == 5

    with pytest.raises(ValueError):
        _ = iterations(0)


def test_iteration_based_progress_progress():
    # --- arrange -----------------------------------------
    tracker_1 = iterations(1)
    tracker_2 = iterations(2)
    tracker_3 = iterations(1000)

    # --- act & assert 1 ----------------------------------
    tracker_1.start()
    tracker_2.start()
    tracker_3.start()
    time.sleep(0.1)  # should not do anything, since these are iteration-based

    assert tracker_1.progress() == Progress(0, 1)
    assert tracker_2.progress() == Progress(0, 2)
    assert tracker_3.progress() == Progress(0, 1000)

    assert tracker_1.progress().is_finished == False
    assert tracker_2.progress().is_finished == False
    assert tracker_3.progress().is_finished == False

    # --- act & assert 2 ----------------------------------
    tracker_1.iteration_done()
    tracker_2.iteration_done()
    tracker_3.iteration_done()

    assert tracker_1.progress() == Progress(1, 1)
    assert tracker_2.progress() == Progress(1, 2)
    assert tracker_3.progress() == Progress(1, 1000)

    assert tracker_1.progress().is_finished == True
    assert tracker_2.progress().is_finished == False
    assert tracker_3.progress().is_finished == False

    # --- act & assert 3 ----------------------------------
    tracker_1.iteration_done()
    tracker_2.iteration_done()
    tracker_3.iteration_done()

    assert tracker_1.progress() == Progress(1, 1)
    assert tracker_2.progress() == Progress(2, 2)
    assert tracker_3.progress() == Progress(2, 1000)

    assert tracker_1.progress().is_finished == True
    assert tracker_2.progress().is_finished == True
    assert tracker_3.progress().is_finished == False

    # --- assert 4 ----------------------------------------
    assert tracker_1.iter_count() == 2
    assert tracker_2.iter_count() == 2
    assert tracker_3.iter_count() == 2

    assert 0.1 <= tracker_1.t_elapsed_sec() <= 1.0
    assert 0.1 <= tracker_2.t_elapsed_sec() <= 1.0
    assert 0.1 <= tracker_3.t_elapsed_sec() <= 1.0


# =================================================================================================
#  TimeBased
# =================================================================================================
def test_time_based_progress_factory_methods():
    assert isinstance(seconds(5), ProgressTracker)
    assert isinstance(minutes(1.5), ProgressTracker)
    assert isinstance(hours(1.25), ProgressTracker)

    assert seconds(5)._max_seconds == 5
    assert minutes(1.5)._max_seconds == 90
    assert hours(1.25)._max_seconds == 4500

    with pytest.raises(ValueError):
        _ = seconds(0)

    with pytest.raises(ValueError):
        _ = minutes(-1)

    with pytest.raises(ValueError):
        _ = hours(-2)


def test_time_based_progress_progress():
    # --- arrange -----------------------------------------
    tracker_1 = seconds(0.001)
    tracker_2 = seconds(0.01)
    tracker_3 = seconds(10)

    # --- act & assert 1 ----------------------------------
    tracker_1.start()
    tracker_2.start()
    tracker_3.start()
    tracker_1.iteration_done()  # should not do anything, since these are time-based
    tracker_2.iteration_done()  # should not do anything, since these are time-based
    tracker_3.iteration_done()  # should not do anything, since these are time-based

    assert tracker_1.progress() == Progress(0, 1)
    assert tracker_2.progress() == Progress(0, 1)
    assert tracker_3.progress() == Progress(0, 10)

    assert tracker_1.progress().is_finished == False
    assert tracker_2.progress().is_finished == False
    assert tracker_3.progress().is_finished == False

    # --- act & assert 2 ----------------------------------
    time.sleep(0.002)

    assert tracker_1.progress() == Progress(1, 1)
    assert tracker_2.progress() == Progress(0, 1)
    assert tracker_3.progress() == Progress(0, 10)

    assert tracker_1.progress().is_finished == True
    assert tracker_2.progress().is_finished == False
    assert tracker_3.progress().is_finished == False

    # --- act & assert 3 ----------------------------------
    time.sleep(0.02)

    assert tracker_1.progress() == Progress(1, 1)
    assert tracker_2.progress() == Progress(1, 1)
    assert tracker_3.progress() == Progress(0, 10)

    assert tracker_1.progress().is_finished == True
    assert tracker_2.progress().is_finished == True
    assert tracker_3.progress().is_finished == False

    # --- assert 4 ----------------------------------------
    assert tracker_1.iter_count() == 1
    assert tracker_2.iter_count() == 1
    assert tracker_3.iter_count() == 1

    assert 0.022 <= tracker_1.t_elapsed_sec() <= 1.0
    assert 0.022 <= tracker_2.t_elapsed_sec() <= 1.0
    assert 0.022 <= tracker_3.t_elapsed_sec() <= 1.0
