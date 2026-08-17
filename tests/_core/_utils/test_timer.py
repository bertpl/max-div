import time

import pytest

from max_div._core._utils import Timer


def test_timer(fake_clock):
    # --- arrange -----------------------------------------
    timer = Timer()

    # --- assert 1 ----------------------------------------
    with pytest.raises(RuntimeError):
        timer.t_elapsed_sec()

    # --- act ---------------------------------------------
    with timer:
        time.sleep(0.1)  # patched by fake_clock: advances the clock by exactly 0.1 s, no real wait

    # --- assert ------------------------------------------
    assert timer.t_elapsed_sec() == pytest.approx(0.1)
    assert timer.t_elapsed_nsec() == pytest.approx(0.1e9)


def test_timer_running(fake_clock):
    # --- arrange -----------------------------------------
    timer = Timer()

    # --- act ---------------------------------------------
    with timer:
        t_before = timer.t_elapsed_sec()
        time.sleep(0.1)
        t_after = timer.t_elapsed_sec()

    # --- assert ------------------------------------------
    assert t_before == pytest.approx(0.0)
    assert t_after == pytest.approx(0.1)  # the clock advanced while the timer was running


def test_timer_measures_the_real_clock():
    """Guard against the fake clock hiding a Timer that reads no clock at all.

    Asserts only a floor: `sleep(0.1)` guarantees a minimum, never a maximum, and how promptly the OS
    reschedules the woken thread is not something Timer promises anything about.
    """
    # --- arrange -----------------------------------------
    timer = Timer()

    # --- act ---------------------------------------------
    with timer:
        time.sleep(0.1)  # the real sleep -- this test does not request fake_clock

    # --- assert ------------------------------------------
    assert timer.t_elapsed_sec() >= 0.09
    assert timer.t_elapsed_nsec() >= 90_000_000
