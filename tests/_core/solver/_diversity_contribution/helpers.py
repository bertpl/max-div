"""Helpers shared by the diversity-contribution tracker tests."""

import numpy as np


def selection_args(indices: list[int], n: int) -> tuple[np.ndarray, np.int32]:
    """Build the (selected, n_selected) argument pair of `contribution_wrt_selection` from the selected indices."""
    selected = np.full(n, False, dtype=np.bool)
    selected[indices] = True
    return selected, np.int32(len(indices))
