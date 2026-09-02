import pytest

from max_div._core.solver._parallel import FixedGroupCount, PowerLawGroupMerge


@pytest.mark.parametrize(
    "n_workers,rate,fraction,expected",
    [
        (12, 1.0, 0.0, 12),
        (12, 1.0, 0.05, 12),
        (12, 1.0, 0.1, 11),
        (12, 1.0, 0.5, 6),
        (12, 1.0, 0.92, 1),  # just past the last merge at 11/12
        (12, 1.0, 1.0, 1),
        (12, 1.0, -0.5, 12),  # a not-yet-started tracker clamps to the start
        (12, 1.0, 1.5, 1),  # an overspent budget clamps to the end
        (4, 1.0, 0.5, 2),
        (1, 1.0, 0.3, 1),
        (12, 2.0, 0.1, 10),  # 12 * 0.81 = 9.72
        (12, 2.0, 0.5, 3),  # 12 * 0.25
        (12, 2.0, 0.7, 2),  # 12 * 0.09 = 1.08
        (12, 2.0, 0.72, 1),  # just past the last merge at 1 - sqrt(1/12) = 0.711
        (4, 2.0, 0.5, 1),  # 4 * 0.25
    ],
)
def test_the_power_law_count_is_the_remaining_progress_power_rounded_up(n_workers, rate, fraction, expected):
    """The count is ceil(n_workers * (1 - fraction) ** rate), at least one, with the fraction clamped to 0..1."""
    # --- act / assert -----------------
    assert PowerLawGroupMerge(n_workers, rate).group_count(fraction) == expected


@pytest.mark.parametrize("fraction", [-1.0, 0.0, 0.5, 1.0, 2.0])
def test_a_fixed_group_count_holds_at_every_fraction(fraction):
    """The fixed schedule returns the configured count whatever the progress, so no merge ever fires."""
    # --- act / assert -----------------
    assert FixedGroupCount(3).group_count(fraction) == 3
