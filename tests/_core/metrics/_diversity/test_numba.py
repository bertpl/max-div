import numpy as np

from max_div._core.metrics._diversity._numba import min_separation


def test_min_separation_ignores_inf_entries():
    """+inf marks an item with no selected neighbor; the minimum skips +inf entries and is +inf when every entry is."""
    # --- arrange ----------------------
    with_sentinels = np.array([np.inf, 2.5, np.inf, 0.75, np.inf], dtype=np.float32)
    all_sentinels = np.full(4, np.inf, dtype=np.float32)

    # --- act / assert -----------------
    assert min_separation(with_sentinels) == np.float32(0.75)
    assert min_separation(all_sentinels) == np.inf
