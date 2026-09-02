"""Distance-sum update for a removal that is only scored and then reverted.

The score reads the selected items only, so the subtraction runs over the selection and leaves
the not-selected sums stale; see the separation counterpart for why the tracker exposes this only
as `remove_trial`, and why it reads distances through the layout-dispatching `get_distance`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray


@numba.njit(numba.void(numba.float64[::1], DISTANCE_STORE_TYPE, numba.int32, numba.int32[::1]), cache=True)
def remove_trial(
    dist_sums: NDArray[np.float64],
    store: DistanceStore,
    i_removed: np.int32,
    new_selection: NDArray[np.int32],
) -> None:
    """Subtract the removed item's distance from the sums of the items in `new_selection`; leave the rest stale."""
    for j in new_selection:
        dist_sums[j] -= np.float64(get_distance(store, i_removed, j))
