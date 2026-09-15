import numpy as np
import pytest

from max_div._core.metrics._diversity._numba import harmonic_mean_separation, min_separation


def test_min_separation_ignores_inf_entries():
    """+inf marks an item with no selected neighbor; the minimum skips +inf entries and is +inf when every entry is."""
    # --- arrange ----------------------
    with_sentinels = np.array([np.inf, 2.5, np.inf, 0.75, np.inf], dtype=np.float32)
    all_sentinels = np.full(4, np.inf, dtype=np.float32)

    # --- act / assert -----------------
    assert min_separation(with_sentinels) == np.float32(0.75)
    assert min_separation(all_sentinels) == np.inf


@pytest.mark.parametrize(
    "sep, expected",
    [
        pytest.param([np.inf, 2.0], 4.0, id="an_inf_entry_adds_nothing_to_the_reciprocal_sum"),
        pytest.param([np.inf, np.inf], np.inf, id="all_inf_entries_give_inf"),
        pytest.param([0.0, np.inf], 0.0, id="a_zero_entry_wins_over_an_inf_entry"),
    ],
)
def test_harmonic_mean_separation_zero_and_inf_limits(sep: list[float], expected: float):
    """A zero separation makes the harmonic mean zero, +inf entries add nothing, and all-+inf gives +inf."""
    # --- act / assert -----------------
    assert harmonic_mean_separation(np.array(sep, dtype=np.float32)) == expected
