import time
from unittest.mock import ANY

import pytest
from tqdm import tqdm

from max_div.solver._duration import (
    Elapsed,
    Progress,
    ProgressTracker,
    TargetDuration,
    _IterationTracker,
    _TimeTracker,
    hours,
    iterations,
    minutes,
    seconds,
)


# =================================================================================================
#  TargetDuration
# =================================================================================================
def test_target_duration_factory_methods():
    assert isinstance(iterations(1), TargetDuration)
    assert isinstance(seconds(1.23), TargetDuration)
    assert isinstance(minutes(2.34), TargetDuration)
    assert isinstance(hours(3.45), TargetDuration)

    with pytest.raises(ValueError):
        _ = iterations(0)

    with pytest.raises(ValueError):
        _ = seconds(0.0)

    with pytest.raises(ValueError):
        _ = minutes(-1.2)

    with pytest.raises(ValueError):
        _ = hours(-0.1)


@pytest.mark.parametrize(
    "duration, expected_repr",
    [
        (iterations(1000), "TargetDuration(1_000 iterations)"),
        (seconds(0.1234), "TargetDuration(0.123 seconds)"),
        (seconds(1.234), "TargetDuration(1.23 seconds)"),
        (seconds(10.5), "TargetDuration(10.5 seconds)"),
        (minutes(5.0), "TargetDuration(300 seconds)"),
        (hours(1.0), "TargetDuration(3_600 seconds)"),
    ],
)
def test_target_duration_str_repr(duration: TargetDuration, expected_repr: str):
    assert str(duration) == expected_repr
    assert repr(duration) == expected_repr


# =================================================================================================
#  ProgressTracker
# =================================================================================================
@pytest.mark.parametrize(
    "duration, expected_tracker_cls",
    [
        (iterations(100), _IterationTracker),
        (seconds(10.0), _TimeTracker),
        (minutes(5.8), _TimeTracker),
        (hours(2.1), _TimeTracker),
    ],
)
def test_progress_tracker_track(duration: TargetDuration, expected_tracker_cls):
    assert isinstance(duration.track(), ProgressTracker)
    assert isinstance(duration.track(), expected_tracker_cls)


def test_progress_tracker_iters_per_second():
    # --- arrange -----------------------------------------
    tracker = seconds(1.0).track()

    # --- act ---------------------------------------------
    ips_1a, ips_1b = tracker.iters_per_second(), tracker.get_progress().est_iters_per_second
    tracker.report_iterations_done(1)
    time.sleep(0.1)
    ips_2a, ips_2b = tracker.iters_per_second(), tracker.get_progress().est_iters_per_second

    # --- assert ------------------------------------------
    assert ips_1a == 0.0  # 0 iterations -> 0 iters/sec
    assert ips_1b == 0.0  # 0 iterations -> 0 iters/sec

    assert 5.0 <= ips_2a <= 20.0  # should be around 10 iters/sec
    assert 5.0 <= ips_2b <= 20.0  # should be around 10 iters/sec


def test_progress_tracker_iteration_based():
    # --- arrange -----------------------------------------
    tracker_1 = iterations(1).track()
    tracker_2 = iterations(5).track()
    tracker_3 = iterations(1000).track()

    # --- assert 0 ----------------------------------------
    assert tracker_1.get_progress().tqdm_n_total == 1
    assert tracker_2.get_progress().tqdm_n_total == 5
    assert tracker_3.get_progress().tqdm_n_total == 1000

    assert tracker_1.get_progress().est_n_iters_remaining == 1
    assert tracker_2.get_progress().est_n_iters_remaining == 5
    assert tracker_3.get_progress().est_n_iters_remaining == 1000

    assert tracker_1.get_progress().est_iters_per_second == 0.0
    assert tracker_2.get_progress().est_iters_per_second == 0.0
    assert tracker_3.get_progress().est_iters_per_second == 0.0

    # --- act & assert 1 ----------------------------------
    time.sleep(0.1)  # should not do anything, since these are iteration-based

    assert tracker_1.get_progress().fraction == 0.0
    assert tracker_2.get_progress().fraction == 0.0
    assert tracker_3.get_progress().fraction == 0.0

    assert tracker_1.get_progress().est_n_iters_remaining == 1  # unchanged
    assert tracker_2.get_progress().est_n_iters_remaining == 5  # unchanged
    assert tracker_3.get_progress().est_n_iters_remaining == 1000  # unchanged

    assert tracker_1.get_progress().est_iters_per_second == 0.0  # unchanged
    assert tracker_2.get_progress().est_iters_per_second == 0.0  # unchanged
    assert tracker_3.get_progress().est_iters_per_second == 0.0  # unchanged

    assert tracker_1.get_progress().is_finished == False
    assert tracker_2.get_progress().is_finished == False
    assert tracker_3.get_progress().is_finished == False

    # --- act & assert 2 ----------------------------------
    tracker_1.report_iterations_done(1)
    tracker_2.report_iterations_done(1)
    tracker_3.report_iterations_done(1)

    assert tracker_1.get_progress().tqdm_n_current == 1
    assert tracker_2.get_progress().tqdm_n_current == 1
    assert tracker_3.get_progress().tqdm_n_current == 1

    assert tracker_1.get_progress().est_n_iters_remaining == 0
    assert tracker_2.get_progress().est_n_iters_remaining == 4
    assert tracker_3.get_progress().est_n_iters_remaining == 999

    assert 5 < tracker_1.get_progress().est_iters_per_second < 15  # should be around 10 iters/sec
    assert 5 < tracker_2.get_progress().est_iters_per_second < 15  # should be around 10 iters/sec
    assert 5 < tracker_3.get_progress().est_iters_per_second < 15  # should be around 10 iters/sec

    assert tracker_1.get_progress().fraction == pytest.approx(1)
    assert tracker_2.get_progress().fraction == pytest.approx(1 / 5)
    assert tracker_3.get_progress().fraction == pytest.approx(1 / 1000)

    assert tracker_1.get_progress().is_finished == True
    assert tracker_2.get_progress().is_finished == False
    assert tracker_3.get_progress().is_finished == False

    # --- act & assert 3 ----------------------------------
    tracker_1.report_iterations_done(4)
    tracker_2.report_iterations_done(4)
    tracker_3.report_iterations_done(4)

    assert tracker_1.get_progress().tqdm_n_current == 1  # clipped to max iters
    assert tracker_2.get_progress().tqdm_n_current == 5
    assert tracker_3.get_progress().tqdm_n_current == 5

    assert tracker_1.get_progress().est_n_iters_remaining == 0
    assert tracker_2.get_progress().est_n_iters_remaining == 0
    assert tracker_3.get_progress().est_n_iters_remaining == 995

    assert tracker_1.get_progress().fraction == pytest.approx(1)
    assert tracker_2.get_progress().fraction == pytest.approx(1)
    assert tracker_3.get_progress().fraction == pytest.approx(5 / 1000)

    assert tracker_1.get_progress().is_finished == True
    assert tracker_2.get_progress().is_finished == True
    assert tracker_3.get_progress().is_finished == False

    # --- assert 4 ----------------------------------------
    elapsed_1 = tracker_1.elapsed()
    elapsed_2 = tracker_2.elapsed()
    elapsed_3 = tracker_3.elapsed()

    assert elapsed_1.n_iterations == 5
    assert elapsed_2.n_iterations == 5
    assert elapsed_3.n_iterations == 5

    assert 0.1 <= elapsed_1.t_elapsed_sec <= 1.0
    assert 0.1 <= elapsed_2.t_elapsed_sec <= 1.0
    assert 0.1 <= elapsed_3.t_elapsed_sec <= 1.0


def test_progress_tracker_time_based():
    # --- arrange -----------------------------------------
    tracker_1 = seconds(0.001).track()
    tracker_2 = seconds(0.01).track()
    tracker_3 = seconds(10).track()

    # --- assert 0 ----------------------------------------
    assert tracker_1.get_progress().tqdm_n_total == 1
    assert tracker_2.get_progress().tqdm_n_total == 1
    assert tracker_3.get_progress().tqdm_n_total == 10

    # --- act & assert 1 ----------------------------------
    tracker_1.report_iterations_done(1)  # should not do anything, since these are time-based
    tracker_2.report_iterations_done(1)  # should not do anything, since these are time-based
    tracker_3.report_iterations_done(1)  # should not do anything, since these are time-based

    assert tracker_1.get_progress().tqdm_n_current == 0
    assert tracker_2.get_progress().tqdm_n_current == 0
    assert tracker_3.get_progress().tqdm_n_current == 0

    assert tracker_1.get_progress().est_n_iters_remaining >= 1
    assert tracker_2.get_progress().est_n_iters_remaining >= 1
    assert tracker_3.get_progress().est_n_iters_remaining >= 1

    assert tracker_1.get_progress().is_finished == False
    assert tracker_2.get_progress().is_finished == False
    assert tracker_3.get_progress().is_finished == False

    # --- act & assert 2 ----------------------------------
    time.sleep(0.002)

    assert tracker_1.get_progress().tqdm_n_current == 1
    assert tracker_2.get_progress().tqdm_n_current == 0
    assert tracker_3.get_progress().tqdm_n_current == 0

    assert tracker_1.get_progress().est_n_iters_remaining == 0
    assert tracker_2.get_progress().est_n_iters_remaining >= 1
    assert tracker_3.get_progress().est_n_iters_remaining >= 1

    assert tracker_1.get_progress().is_finished == True
    assert tracker_2.get_progress().is_finished == False
    assert tracker_3.get_progress().is_finished == False

    # --- act & assert 3 ----------------------------------
    time.sleep(0.02)

    assert tracker_1.get_progress().tqdm_n_current == 1
    assert tracker_2.get_progress().tqdm_n_current == 1
    assert tracker_3.get_progress().tqdm_n_current == 0

    assert tracker_1.get_progress().est_n_iters_remaining == 0
    assert tracker_2.get_progress().est_n_iters_remaining == 0
    assert tracker_3.get_progress().est_n_iters_remaining >= 1

    assert tracker_1.get_progress().is_finished == True
    assert tracker_2.get_progress().is_finished == True
    assert tracker_3.get_progress().is_finished == False

    # --- assert 4 ----------------------------------------
    elapsed_1 = tracker_1.elapsed()
    elapsed_2 = tracker_2.elapsed()
    elapsed_3 = tracker_3.elapsed()

    assert elapsed_1.n_iterations == 1
    assert elapsed_2.n_iterations == 1
    assert elapsed_3.n_iterations == 1

    assert 0.022 <= elapsed_1.t_elapsed_sec <= 1.0
    assert 0.022 <= elapsed_2.t_elapsed_sec <= 1.0
    assert 0.022 <= elapsed_3.t_elapsed_sec <= 1.0


@pytest.mark.parametrize(
    "duration,sleep_time,expected_n_current,expected_n_total",
    [
        (seconds(0.1), 0.2, 1, 1),
        (seconds(0.9), 0.0, 0, 1),
        (seconds(0.8), 0.5, 0, 1),
        (seconds(1.7), 1.1, 0, 1),
    ],
)
def test_progress_corner_cases(
    duration: TargetDuration, sleep_time: float, expected_n_current: int, expected_n_total: int
):
    """Test corner cases where rounding could (but won't) result in n_current == n_total, while we're not finished."""
    # --- arrange -----------------------------------------
    tracker = duration.track()
    time.sleep(sleep_time)

    # --- act ---------------------------------------------
    progress = tracker.get_progress()

    # --- assert ------------------------------------------
    assert progress.tqdm_n_current == expected_n_current
    assert progress.tqdm_n_total == expected_n_total


# =================================================================================================
#  Progress
# =================================================================================================
def test_progress_is_finished():
    dummy_kwargs = dict(iter_count=ANY, est_n_iters_remaining=ANY, est_iters_per_second=ANY)
    progress_1 = Progress(tqdm_n_total=5, fraction=0.0, **dummy_kwargs)
    progress_2 = Progress(tqdm_n_total=5, fraction=0.999, **dummy_kwargs)
    progress_3 = Progress(tqdm_n_total=5, fraction=1.0, **dummy_kwargs)
    progress_4 = Progress(tqdm_n_total=5, fraction=1.1, **dummy_kwargs)

    assert progress_1.is_finished == False
    assert progress_2.is_finished == False
    assert progress_3.is_finished == True
    assert progress_4.is_finished == True


def test_progress_update_tqdm():
    # --- arrange -----------------------------------------
    pbar = tqdm("description")
    progress = Progress(
        fraction=0.3,
        tqdm_n_total=10,
        iter_count=ANY,
        est_n_iters_remaining=ANY,
        est_iters_per_second=ANY,
    )

    # --- act ---------------------------------------------
    progress.update_tqdm(pbar)

    # --- assert ------------------------------------------
    assert pbar.n == progress.tqdm_n_current
    assert pbar.total == progress.tqdm_n_total


# =================================================================================================
#  Elapsed
# =================================================================================================
def test_elapsed_math():
    # --- arrange -----------------------------------------
    d1 = Elapsed(1.0, 10)
    d2 = Elapsed(2.5, 15)
    d3 = Elapsed(0.2, 3)

    # --- act ---------------------------------------------
    s1a = d1 + 0
    s1b = 0.0 + d1
    s12 = d1 + d2
    s23 = d2 + d3
    s123 = sum([d1, d2, d3])

    with pytest.raises(TypeError):
        _ = d1 + "invalid"

    # --- assert ------------------------------------------
    assert s1a == d1
    assert s1b == d1
    assert s12 == Elapsed(3.5, 25)
    assert s23 == Elapsed(2.7, 18)
    assert s123 == Elapsed(3.7, 28)
