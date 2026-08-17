import time

import pytest

from max_div._core._utils import Timer


def test_timer():
    # --- arrange -----------------------------------------
    timer = Timer()

    # --- assert 1 ----------------------------------------
    with pytest.raises(RuntimeError):
        timer.t_elapsed_sec()

    # --- act ---------------------------------------------
    with timer:
        time.sleep(0.1)

    # --- assert ------------------------------------------
    # `sleep(0.1)` blocks for at least 0.1 s and a loaded runner only overshoots, so each reading is
    # bounded from below but not above; an upper bound would eventually flake. The nsec bound is
    # asserted directly (not derived from the sec one) so it also catches a wrong time unit.
    assert timer.t_elapsed_sec() >= 0.09, "t_elapsed_sec() undercounts a 0.1 s sleep."
    assert timer.t_elapsed_nsec() >= 90_000_000, "t_elapsed_nsec() undercounts a 0.1 s sleep."


def test_timer_running():
    # --- arrange -----------------------------------------
    timer = Timer()

    # --- act ---------------------------------------------
    with timer:
        t_before = timer.t_elapsed_sec()
        time.sleep(0.1)
        t_after = timer.t_elapsed_sec()

    # --- assert ------------------------------------------
    assert t_after >= t_before + 0.09, "t_elapsed_sec() did not increase while timer was running."
