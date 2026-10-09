import inspect

import pytest

from max_div._core._warnings import DistanceInputWarning, warn_at_first_frame_outside_max_div


def test_warn_at_first_frame_outside_max_div_points_at_the_calling_line():
    """Called from outside max-div, the warning is attributed to the caller's file and line."""
    # --- act --------------------------
    calling_line = inspect.currentframe().f_lineno + 2  # the line of the call inside the block below
    with pytest.warns(DistanceInputWarning) as record:
        warn_at_first_frame_outside_max_div("message", DistanceInputWarning)

    # --- assert -----------------------
    assert (record[0].filename, record[0].lineno) == (__file__, calling_line)
