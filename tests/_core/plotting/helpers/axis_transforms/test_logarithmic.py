import math

import numpy as np

from max_div._core.plotting.helpers import LogTransform


def test_log_transform_scalar_round_trip() -> None:
    # --- act --------------------------
    to_axis = LogTransform().to_axis(math.e)
    from_axis = LogTransform().from_axis(to_axis)

    # --- assert -----------------------
    assert to_axis == 1.0 and isinstance(to_axis, float)
    assert from_axis == math.e and isinstance(from_axis, float)


def test_log_transform_array_round_trip() -> None:
    # --- arrange ----------------------
    x = np.array([1.0, 10.0, 100.0])

    # --- act --------------------------
    to_axis = LogTransform().to_axis(x)
    from_axis = LogTransform().from_axis(to_axis)

    # --- assert -----------------------
    np.testing.assert_allclose(to_axis, np.log(x))
    np.testing.assert_allclose(from_axis, x)
