import pytest

from max_div.internal.benchmarking._platform_speed import estimate_platform_speed


@pytest.mark.parametrize("silent", [True, False])
def test_estimate_platform_speed(silent: bool):
    # --- act ---------------------------------------------
    s = estimate_platform_speed(silent=silent, fast=True)  # fast=True covers all code; but is >>> faster to test

    # --- assert ------------------------------------------
    assert 1e-2 <= s <= 1e2  # result should be within a factor 100x either side of the reference platform
