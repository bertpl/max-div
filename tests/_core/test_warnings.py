import inspect

import pytest

from max_div._core._warnings import DistanceInputWarning, warn_outside_max_div


def test_warn_outside_max_div_points_at_the_calling_line():
    """Called from outside max-div, the warning is attributed to the caller's file and line."""
    # --- act --------------------------
    with pytest.warns(DistanceInputWarning) as record:
        calling_line = inspect.currentframe().f_lineno + 1
        warn_outside_max_div("message", DistanceInputWarning)

    # --- assert -----------------------
    assert (record[0].filename, record[0].lineno) == (__file__, calling_line)
