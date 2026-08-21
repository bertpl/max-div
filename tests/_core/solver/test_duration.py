import multiprocessing
import time
from multiprocessing.queues import Queue
from unittest.mock import ANY

import pytest
from tqdm import tqdm

from max_div._core.solver._duration import (
    Elapsed,
    Progress,
    ProgressTracker,
    TargetDuration,
    TargetTotalTimeDuration,
    _IterationTracker,
    _TimeTracker,
    hours,
    iterations,
    minutes,
    seconds,
    total_hours,
    total_minutes,
    total_seconds,
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
    "duration, expected_value",
    [
        (iterations(1000), 1000.0),
        (seconds(0.1234), 0.1234),
        (minutes(1.0), 60.0),
    ],
)
def test_target_duration_value(duration: TargetDuration, expected_value: float):
    assert duration.value() == expected_value


@pytest.mark.parametrize(
    "duration_a, duration_b, expected_equal",
    [
        (iterations(1000), iterations(1000), True),
        (iterations(1000), iterations(999), False),
        (seconds(1.0), seconds(1.0), True),
        (seconds(30.0), minutes(0.5), True),
        (seconds(1), minutes(1), False),
        (seconds(1), iterations(1), False),
    ],
)
def test_target_duration_eq(duration_a: TargetDuration, duration_b: TargetDuration, expected_equal: bool):
    # --- act --------------------------
    is_equal = duration_a == duration_b
    is_not_equal = duration_a != duration_b

    # --- assert -----------------------
    assert is_equal is expected_equal
    assert is_not_equal is (not expected_equal)


@pytest.mark.parametrize(
    "duration, expected_repr, expected_str",
    [
        (iterations(1000), "TargetDuration(1_000 it.)", "1_000 it."),
        (seconds(0.1234), "TargetDuration(123.40ms)", "123.40ms"),
        (seconds(1.234), "TargetDuration(1.23s)", "1.23s"),
        (seconds(10.5), "TargetDuration(10.50s)", "10.50s"),
        (minutes(5.0), "TargetDuration(5m0.00s)", "5m0.00s"),
        (hours(1.0), "TargetDuration(1h0m0.0s)", "1h0m0.0s"),
    ],
)
def test_target_duration_repr_str(duration: TargetDuration, expected_repr: str, expected_str: str):
    assert repr(duration) == expected_repr
    assert str(duration) == expected_str


@pytest.mark.parametrize(
    "duration, multiplier, expected_duration",
    [
        (iterations(100), 2, iterations(200)),
        (iterations(150), 0.5, iterations(75)),
        (iterations(150), 0.001, iterations(1)),
        (seconds(10.0), 3, seconds(30.0)),
        (minutes(4.0), 0.5, minutes(2.0)),
        (seconds(1.0), 1e-15, seconds(1e-9)),
    ],
)
def test_target_duration_mul(duration: TargetDuration, multiplier: float | int, expected_duration: TargetDuration):
    # --- act --------------------------
    result_duration_1 = duration * multiplier
    result_duration_2 = multiplier * duration

    # --- assert -----------------------

    # check type
    assert type(result_duration_1) is type(expected_duration)
    assert type(result_duration_2) is type(expected_duration)

    # check equality (roughly) in type-independent way
    assert str(result_duration_1) == str(expected_duration)
    assert str(result_duration_2) == str(expected_duration)


def test_target_duration_mul_type_error():
    # --- act & assert -----------------
    with pytest.raises(TypeError):
        _ = iterations(100) * "invalid"

    with pytest.raises(TypeError):
        _ = "invalid" * iterations(100)

    with pytest.raises(TypeError):
        _ = seconds(60.0) * "invalid"

    with pytest.raises(TypeError):
        _ = "invalid" * seconds(60.0)


def test_duration_lt():
    assert iterations(100) < iterations(200)
    assert seconds(10.0) < seconds(20.0)
    assert seconds(30.0) < minutes(1.0)
    assert minutes(2.0) < hours(0.1)

    with pytest.raises(TypeError):
        _ = iterations(100) < "bla"


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


def test_progress_tracker_iters_per_second(fake_clock):
    # --- arrange ----------------------
    tracker = seconds(1.0).track()

    # --- act --------------------------
    ips_1a, ips_1b = tracker.iters_per_second(), tracker.get_progress().est_iters_per_second
    tracker.report_iterations_done(1)
    fake_clock.advance(0.1)
    ips_2a, ips_2b = tracker.iters_per_second(), tracker.get_progress().est_iters_per_second

    # --- assert -----------------------
    assert ips_1a == 0.0  # 0 iterations -> 0 iters/sec
    assert ips_1b == 0.0  # 0 iterations -> 0 iters/sec

    assert ips_2a == pytest.approx(10.0)  # 1 iteration over exactly 0.1 s
    assert ips_2b == pytest.approx(10.0)


def test_progress_tracker_iteration_based(fake_clock):
    # --- arrange ----------------------
    tracker_1 = iterations(1).track()
    tracker_2 = iterations(5).track()
    tracker_3 = iterations(1000).track()

    # --- assert 0 ---------------------
    assert tracker_1.get_progress().tqdm_n_total == 1
    assert tracker_2.get_progress().tqdm_n_total == 5
    assert tracker_3.get_progress().tqdm_n_total == 1000

    assert tracker_1.get_progress().est_n_iters_remaining == 1
    assert tracker_2.get_progress().est_n_iters_remaining == 5
    assert tracker_3.get_progress().est_n_iters_remaining == 1000

    assert tracker_1.get_progress().est_iters_per_second == 0.0
    assert tracker_2.get_progress().est_iters_per_second == 0.0
    assert tracker_3.get_progress().est_iters_per_second == 0.0

    # --- act & assert 1 ---------------
    fake_clock.advance(0.1)  # time passing should not move iteration-based progress

    assert tracker_1.get_progress().fraction == 0.0
    assert tracker_2.get_progress().fraction == 0.0
    assert tracker_3.get_progress().fraction == 0.0

    assert tracker_1.get_progress().est_n_iters_remaining == 1  # unchanged
    assert tracker_2.get_progress().est_n_iters_remaining == 5  # unchanged
    assert tracker_3.get_progress().est_n_iters_remaining == 1000  # unchanged

    assert tracker_1.get_progress().est_iters_per_second == 0.0  # unchanged
    assert tracker_2.get_progress().est_iters_per_second == 0.0  # unchanged
    assert tracker_3.get_progress().est_iters_per_second == 0.0  # unchanged

    assert not tracker_1.get_progress().is_finished
    assert not tracker_2.get_progress().is_finished
    assert not tracker_3.get_progress().is_finished

    # --- act & assert 2 ---------------
    tracker_1.report_iterations_done(1)
    tracker_2.report_iterations_done(1)
    tracker_3.report_iterations_done(1)

    assert tracker_1.get_progress().tqdm_n_current == 1
    assert tracker_2.get_progress().tqdm_n_current == 1
    assert tracker_3.get_progress().tqdm_n_current == 1

    assert tracker_1.get_progress().est_n_iters_remaining == 0
    assert tracker_2.get_progress().est_n_iters_remaining == 4
    assert tracker_3.get_progress().est_n_iters_remaining == 999

    assert tracker_1.get_progress().est_iters_per_second == pytest.approx(10.0)  # 1 iter / 0.1 s
    assert tracker_2.get_progress().est_iters_per_second == pytest.approx(10.0)
    assert tracker_3.get_progress().est_iters_per_second == pytest.approx(10.0)

    assert tracker_1.get_progress().fraction == pytest.approx(1)
    assert tracker_2.get_progress().fraction == pytest.approx(1 / 5)
    assert tracker_3.get_progress().fraction == pytest.approx(1 / 1000)

    assert tracker_1.get_progress().is_finished
    assert not tracker_2.get_progress().is_finished
    assert not tracker_3.get_progress().is_finished

    # --- act & assert 3 ---------------
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

    assert tracker_1.get_progress().is_finished
    assert tracker_2.get_progress().is_finished
    assert not tracker_3.get_progress().is_finished

    # --- assert 4 ---------------------
    elapsed_1 = tracker_1.elapsed()
    elapsed_2 = tracker_2.elapsed()
    elapsed_3 = tracker_3.elapsed()

    assert elapsed_1.n_iterations == 5
    assert elapsed_2.n_iterations == 5
    assert elapsed_3.n_iterations == 5

    assert elapsed_1.t_elapsed_sec == pytest.approx(0.1)
    assert elapsed_2.t_elapsed_sec == pytest.approx(0.1)
    assert elapsed_3.t_elapsed_sec == pytest.approx(0.1)


def test_progress_tracker_time_based(fake_clock):
    # --- arrange ----------------------
    tracker_1 = seconds(0.001).track()
    tracker_2 = seconds(0.01).track()
    tracker_3 = seconds(10).track()

    # --- assert 0 ---------------------
    assert tracker_1.get_progress().tqdm_n_total == 1
    assert tracker_2.get_progress().tqdm_n_total == 1
    assert tracker_3.get_progress().tqdm_n_total == 10

    # --- act & assert 1 ---------------
    tracker_1.report_iterations_done(1)  # should not do anything, since these are time-based
    tracker_2.report_iterations_done(1)  # should not do anything, since these are time-based
    tracker_3.report_iterations_done(1)  # should not do anything, since these are time-based

    assert tracker_1.get_progress().tqdm_n_current == 0
    assert tracker_2.get_progress().tqdm_n_current == 0
    assert tracker_3.get_progress().tqdm_n_current == 0

    assert tracker_1.get_progress().est_n_iters_remaining >= 1
    assert tracker_2.get_progress().est_n_iters_remaining >= 1
    assert tracker_3.get_progress().est_n_iters_remaining >= 1

    assert not tracker_1.get_progress().is_finished
    assert not tracker_2.get_progress().is_finished
    assert not tracker_3.get_progress().is_finished

    # --- act & assert 2 ---------------
    fake_clock.advance(0.005)  # past tracker_1's 0.001 s budget, still short of tracker_2's 0.01 s

    assert tracker_1.get_progress().tqdm_n_current == 1
    assert tracker_2.get_progress().tqdm_n_current == 0
    assert tracker_3.get_progress().tqdm_n_current == 0

    assert tracker_1.get_progress().est_n_iters_remaining == 0
    assert tracker_2.get_progress().est_n_iters_remaining >= 1
    assert tracker_3.get_progress().est_n_iters_remaining >= 1

    assert tracker_1.get_progress().is_finished
    assert not tracker_2.get_progress().is_finished
    assert not tracker_3.get_progress().is_finished

    # --- act & assert 3 ---------------
    fake_clock.advance(0.015)  # total 0.02 s: past tracker_2's 0.01 s budget, still short of tracker_3's

    assert tracker_1.get_progress().tqdm_n_current == 1
    assert tracker_2.get_progress().tqdm_n_current == 1
    assert tracker_3.get_progress().tqdm_n_current == 0

    assert tracker_1.get_progress().est_n_iters_remaining == 0
    assert tracker_2.get_progress().est_n_iters_remaining == 0
    assert tracker_3.get_progress().est_n_iters_remaining >= 1

    assert tracker_1.get_progress().is_finished
    assert tracker_2.get_progress().is_finished
    assert not tracker_3.get_progress().is_finished

    # --- assert 4 ---------------------
    elapsed_1 = tracker_1.elapsed()
    elapsed_2 = tracker_2.elapsed()
    elapsed_3 = tracker_3.elapsed()

    assert elapsed_1.n_iterations == 1
    assert elapsed_2.n_iterations == 1
    assert elapsed_3.n_iterations == 1

    assert elapsed_1.t_elapsed_sec == pytest.approx(0.02)
    assert elapsed_2.t_elapsed_sec == pytest.approx(0.02)
    assert elapsed_3.t_elapsed_sec == pytest.approx(0.02)


@pytest.mark.parametrize(
    "duration,elapsed_sec,expected_n_current,expected_n_total",
    [
        (seconds(0.1), 0.2, 1, 1),
        (seconds(0.9), 0.0, 0, 1),
        (seconds(0.8), 0.5, 0, 1),
        (seconds(1.7), 1.1, 0, 1),
    ],
)
def test_progress_corner_cases(
    fake_clock, duration: TargetDuration, elapsed_sec: float, expected_n_current: int, expected_n_total: int
):
    """Test corner cases where rounding could (but won't) result in n_current == n_total, while we're not finished."""
    # --- arrange ----------------------
    tracker = duration.track()
    fake_clock.advance(elapsed_sec)

    # --- act --------------------------
    progress = tracker.get_progress()

    # --- assert -----------------------
    assert progress.tqdm_n_current == expected_n_current
    assert progress.tqdm_n_total == expected_n_total


# =================================================================================================
#  Progress
# =================================================================================================
def test_progress_is_finished():
    dummy_kwargs = {"iter_count": ANY, "est_n_iters_remaining": ANY, "est_iters_per_second": ANY}
    progress_1 = Progress(tqdm_n_total=5, fraction=0.0, **dummy_kwargs)
    progress_2 = Progress(tqdm_n_total=5, fraction=0.999, **dummy_kwargs)
    progress_3 = Progress(tqdm_n_total=5, fraction=1.0, **dummy_kwargs)
    progress_4 = Progress(tqdm_n_total=5, fraction=1.1, **dummy_kwargs)

    assert not progress_1.is_finished
    assert not progress_2.is_finished
    assert progress_3.is_finished
    assert progress_4.is_finished


def test_progress_update_tqdm():
    # --- arrange ----------------------
    pbar = tqdm("description")
    progress = Progress(
        fraction=0.3,
        tqdm_n_total=10,
        iter_count=ANY,
        est_n_iters_remaining=ANY,
        est_iters_per_second=ANY,
    )

    # --- act --------------------------
    progress.update_tqdm(pbar)

    # --- assert -----------------------
    assert pbar.n == progress.tqdm_n_current
    assert pbar.total == progress.tqdm_n_total


# =================================================================================================
#  Elapsed
# =================================================================================================
@pytest.mark.parametrize(
    "elapsed, expected_str",
    [
        (Elapsed(t_elapsed_sec=2.17, n_iterations=14854), "2.17s (14_854 iterations)"),
        (Elapsed(t_elapsed_sec=0.005, n_iterations=42), "5.00ms (42 iterations)"),
        (Elapsed(t_elapsed_sec=65.3, n_iterations=100000), "1m5.3s (100_000 iterations)"),
        (Elapsed(t_elapsed_sec=0.0, n_iterations=0), "0.00ns (0 iterations)"),
    ],
)
def test_elapsed_str(elapsed: Elapsed, expected_str: str):
    # --- act --------------------------
    result = str(elapsed)

    # --- assert -----------------------
    assert result == expected_str


def test_elapsed_math():
    # --- arrange ----------------------
    d1 = Elapsed(1.0, 10)
    d2 = Elapsed(2.5, 15)
    d3 = Elapsed(0.2, 3)

    # --- act --------------------------
    s1a = d1 + 0
    s1b = 0.0 + d1
    s12 = d1 + d2
    s23 = d2 + d3
    s123 = sum([d1, d2, d3])

    with pytest.raises(TypeError):
        _ = d1 + "invalid"

    # --- assert -----------------------
    assert s1a == d1
    assert s1b == d1
    assert s12 == Elapsed(3.5, 25)
    assert s23 == Elapsed(2.7, 18)
    assert s123 == Elapsed(3.7, 28)


# =================================================================================================
#  TargetTotalTimeDuration
# =================================================================================================
def test_total_duration_factory_methods():
    """The total factories build total budgets, in the same three units as the step-budget ones."""
    assert isinstance(total_seconds(1.23), TargetTotalTimeDuration)
    assert isinstance(total_minutes(2.34), TargetTotalTimeDuration)
    assert isinstance(total_hours(3.45), TargetTotalTimeDuration)

    assert total_minutes(2.0).value() == pytest.approx(120.0)
    assert total_hours(0.5).value() == pytest.approx(1800.0)

    with pytest.raises(ValueError):
        _ = total_seconds(0.0)


def test_a_total_budget_is_distinct_from_a_step_budget_of_the_same_length():
    """The two answer different questions, so neither equality nor a shared hash may merge them."""
    # --- act / assert -----------------
    assert total_seconds(5.0) != seconds(5.0)
    assert str(total_seconds(5.0)) != str(seconds(5.0))


def test_a_total_budget_counts_down_from_the_moment_it_starts(fake_clock):
    """Constructing a total budget starts it, so a hand-assembled solver still gets sane behavior."""
    # --- arrange ----------------------
    duration = total_seconds(10.0)

    # --- act --------------------------
    fake_clock.advance(4.0)

    # --- assert -----------------------
    assert duration.remaining_seconds() == pytest.approx(6.0)


def test_starting_the_clock_hands_back_the_whole_budget(fake_clock):
    """The solver anchors a copy when it starts working, leaving the caller's budget untouched."""
    # --- arrange ----------------------
    duration = total_seconds(10.0)
    fake_clock.advance(4.0)

    # --- act --------------------------
    anchored = duration.started_now()

    # --- assert -----------------------
    assert anchored is not duration
    assert anchored.remaining_seconds() == pytest.approx(10.0)
    assert duration.remaining_seconds() == pytest.approx(6.0)


def test_a_step_budget_starts_now_as_itself():
    """Only a total budget has an anchor to move, so the others hand back the very same object."""
    # --- arrange ----------------------
    step_budget = seconds(10.0)
    iteration_budget = iterations(10)

    # --- act / assert -----------------
    assert step_budget.started_now() is step_budget
    assert iteration_budget.started_now() is iteration_budget


def test_a_spent_total_budget_tracks_as_finished_right_away(fake_clock):
    """A solve whose build outlasts the budget skips its optimization instead of overrunning it."""
    # --- arrange ----------------------
    duration = total_seconds(10.0)

    # --- act --------------------------
    fake_clock.advance(11.0)

    # --- assert -----------------------
    assert duration.remaining_seconds() == 0.0
    assert duration.track().get_progress().is_finished


def test_a_scaled_total_budget_keeps_its_type_and_its_start(fake_clock):
    """Scaling stays within the type and does not hand back time the original already spent."""
    # --- arrange ----------------------
    duration = total_seconds(10.0)
    fake_clock.advance(4.0)

    # --- act --------------------------
    scaled = duration * 2.0

    # --- assert -----------------------
    assert isinstance(scaled, TargetTotalTimeDuration)
    assert scaled.value() == pytest.approx(20.0)
    # started 4s ago and now worth 20s, so 16s is left -- not the full 20s of a fresh budget
    assert scaled.remaining_seconds() == pytest.approx(16.0)


def test_a_total_budget_cannot_be_scaled_by_a_non_number():
    """The refusal to scale must surface, rather than silently producing some other type."""
    # --- arrange / act / assert -------
    with pytest.raises(TypeError):
        _ = total_seconds(10.0) * "invalid"


def test_a_deadline_means_the_same_thing_in_a_spawned_process():
    """A portfolio's workers read the parent's deadline, so the clock behind it has to be machine-wide."""
    # --- arrange ----------------------
    context = multiprocessing.get_context("spawn")
    results: Queue = context.Queue()

    # --- act --------------------------
    parent_now = time.monotonic()
    worker = context.Process(target=_report_monotonic_offset, args=(results, parent_now))
    worker.start()
    offset = results.get(timeout=60)
    worker.join()

    # --- assert -----------------------
    # the offset is the spawn itself; an unrelated clock base would show up as a wild number
    assert 0.0 <= offset < 30.0


def _report_monotonic_offset(results: Queue, parent_now: float) -> None:
    """Report how far this process's monotonic clock reads past the parent's; a spawned worker entry point."""
    results.put(time.monotonic() - parent_now)
