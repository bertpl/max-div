import math

import numpy as np
import pytest

from tests._extras import skip_module_unless_extra

skip_module_unless_extra("plot")

from max_div._core.plotting.helpers import UpperLogTransform  # noqa: E402
from max_div._core.plotting.helpers.axis_transforms.upper_logarithmic import (  # noqa: E402
    _add_ticks_at_precision_jumps,
    _without_trailing_zero,
)

# These values approach 0.8 from below, each ten times closer than the last
SATURATING = [0, 0.5, 0.7, 0.79, 0.799, 0.7999, 0.79999, 0.799999, 0.7999999]


# ==================================================================================================
#  to_axis / from_axis
# ==================================================================================================
def test_to_axis_and_from_axis_follow_the_formulas() -> None:
    # --- arrange ----------------------
    transform = UpperLogTransform(c0=0.5, c1=-2.0, x_ref=10.0)

    # --- act --------------------------
    to_axis = transform.to_axis(9.0)
    from_axis = transform.from_axis(to_axis)

    # --- assert -----------------------
    assert to_axis == 0.5 - 2.0 * math.log(1.0)
    assert from_axis == pytest.approx(9.0)
    assert isinstance(to_axis, float)
    assert isinstance(from_axis, float)


def test_array_round_trip() -> None:
    # --- arrange ----------------------
    transform = UpperLogTransform(c0=0.5, c1=-2.0, x_ref=10.0)
    x = np.array([1.0, 5.0, 9.9])

    # --- act --------------------------
    from_axis = transform.from_axis(transform.to_axis(x))

    # --- assert -----------------------
    np.testing.assert_allclose(from_axis, x)


# ==================================================================================================
#  from_values
# ==================================================================================================
def test_from_values_maps_the_range_onto_unit_interval_above_the_data() -> None:
    # --- act --------------------------
    transform = UpperLogTransform.from_values(SATURATING)

    # --- assert -----------------------
    assert transform.x_ref > max(SATURATING)
    assert transform.c1 < 0  # the axis increases with x
    assert transform.to_axis(min(SATURATING)) == pytest.approx(0.0)
    assert transform.to_axis(max(SATURATING)) == pytest.approx(1.0)


def test_from_values_spreads_saturating_data_evenly() -> None:
    # --- act --------------------------
    x_axis = np.sort(UpperLogTransform.from_values(SATURATING).to_axis(SATURATING))

    # --- assert -----------------------
    # from 0.7 on each value is ten times closer to the limit than the last, so those land at equal steps
    steps = np.diff(x_axis)[2:-2]
    np.testing.assert_allclose(steps, steps.mean(), atol=0.01)


def test_from_values_constant_data_draws_the_value_mid_axis() -> None:
    # --- act --------------------------
    transform = UpperLogTransform.from_values([3.0, 3.0, 3.0])

    # --- assert -----------------------
    assert transform.x_ref == 4.0
    assert transform.to_axis(3.0) == 0.5
    assert transform.c1 < 0


# ==================================================================================================
#  ticks_and_labels
# ==================================================================================================
def test_ticks_and_labels_show_the_precision_each_decade_needs() -> None:
    # --- arrange ----------------------
    transform = UpperLogTransform.from_values(SATURATING)

    # --- act --------------------------
    ticks, labels = transform.ticks_and_labels(n_max=10)

    # --- assert -----------------------
    assert labels == ["0", "0.6", "0.7", "0.78", "0.79", "0.799", "0.7998", "0.7999", "0.79999", "0.799999"]
    assert ticks == [float(label) for label in labels]


@pytest.mark.parametrize("n_max", [3, 5, 10, 20])
def test_ticks_and_labels_never_exceed_n_max(n_max: int) -> None:
    # --- act --------------------------
    ticks, labels = UpperLogTransform.from_values(SATURATING).ticks_and_labels(n_max=n_max)

    # --- assert -----------------------
    assert 2 <= len(ticks) <= n_max
    assert ticks == sorted(ticks)
    assert len(set(labels)) == len(labels)


def test_ticks_and_labels_fill_precision_jumps_with_unlabeled_ticks() -> None:
    # --- arrange ----------------------
    transform = UpperLogTransform.from_values([1.0, 1.4, 1.41, 1.414, 1.4142, 1.41421])

    # --- act --------------------------
    ticks, labels = transform.ticks_and_labels(n_max=8, should_add_missing_ticks=True)

    # --- assert -----------------------
    # between 1.41 and 1.413 the precision jumps from two to three decimals: 1.411 and 1.412 fill the gap
    i_141 = labels.index("1.41")
    assert labels[i_141 : i_141 + 4] == ["1.41", "", "", "1.413"]
    np.testing.assert_allclose(ticks[i_141 : i_141 + 4], [1.41, 1.411, 1.412, 1.413])
    assert [label for label in labels if label] == ["1", "1.3", "1.4", "1.41", "1.413", "1.414", "1.4141", "1.41420"]


def test_ticks_and_labels_recover_when_coarse_rounding_empties_the_grid() -> None:
    # --- arrange ----------------------
    # uniform data on a short range: at 5 ticks, rounding to whole units leaves no label on the axis
    transform = UpperLogTransform.from_values(np.random.default_rng(0).uniform(0.2, 0.9, 50))

    # --- act --------------------------
    ticks, labels = transform.ticks_and_labels(n_max=5)

    # --- assert -----------------------
    assert 2 <= len(ticks) <= 5
    assert labels[0] != ""


def test_ticks_and_labels_keep_the_best_fit_when_the_count_overshoots() -> None:
    # --- arrange ----------------------
    # uniform data: raising the grid from 7 distinct labels jumps straight past twice n_max, so the
    # search stops and keeps the 7
    transform = UpperLogTransform.from_values(np.random.default_rng(0).uniform(0.2, 0.9, 50))

    # --- act --------------------------
    _, labels = transform.ticks_and_labels(n_max=8)

    # --- assert -----------------------
    assert labels == ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]


# ==================================================================================================
#  helpers
# ==================================================================================================
@pytest.mark.parametrize(
    "label, expected",
    [("10", "10"), ("1.0", "1"), ("1.50", "1.5"), ("1.500", "1.50"), ("1.25", "1.25")],
)
def test_without_trailing_zero(label: str, expected: str) -> None:
    assert _without_trailing_zero(label) == expected


def test_add_ticks_at_precision_jumps_leaves_equal_precision_alone() -> None:
    # --- act --------------------------
    ticks, labels = _add_ticks_at_precision_jumps([0.1, 0.2, 0.3], ["0.1", "0.2", "0.3"])

    # --- assert -----------------------
    assert ticks == [0.1, 0.2, 0.3]
    assert labels == ["0.1", "0.2", "0.3"]
