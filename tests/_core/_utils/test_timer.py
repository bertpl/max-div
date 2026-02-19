import time

import pytest

from max_div._core._utils import Timer


@pytest.mark.flaky(reruns=10)
def test_timer():
    # --- arrange -----------------------------------------
    timer = Timer()

    # --- assert 1 ----------------------------------------
    with pytest.raises(RuntimeError):
        t = timer.t_elapsed_sec()

    # --- act ---------------------------------------------
    with timer:
        time.sleep(0.1)

    # --- assert ------------------------------------------
    assert 0.05 < timer.t_elapsed_sec() < 0.15, "t_elapsed_sec() result incorrect."
    assert 50_000_000 < timer.t_elapsed_nsec() < 150_000_000, "t_elapsed_nsec() result incorrect."


@pytest.mark.flaky(reruns=10)
def test_timer_running():
    # --- arrange -----------------------------------------
    timer = Timer()

    # --- act ---------------------------------------------
    with timer as t:
        t_before = timer.t_elapsed_sec()
        time.sleep(0.1)
        t_after = timer.t_elapsed_sec()

    # --- assert ------------------------------------------
    assert t_after >= t_before + 0.05, "t_elapsed_sec() did not increase while timer was running."
