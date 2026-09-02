"""Separation update for a removal that is only scored and then reverted.

The score reads the separations of the selected items only, so a removal made just to score it
needs no update of the not-selected entries; skipping them turns an O(n) sweep into a loop over
the selection.  The not-selected entries are then stale, which is why the tracker exposes this
only as `remove_trial`, and the solver state only inside a scope that always restores.

Unlike the backend modules, this loop reads distances through the layout-dispatching
`get_distance`: it visits the selected items only, in scattered reads, so the layout branch that
would keep a sweep over all items scalar costs nothing measurable here, and one function replaces
one per layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance

from ._signatures import REMOVE_SIGNATURE

if TYPE_CHECKING:
    from numpy.typing import NDArray


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int32, numba.int32[::1]), inline="always", cache=True)
def _nearest_selected(store: DistanceStore, j: np.int32, selection: NDArray[np.int32]) -> np.float32:
    """Return the distance from item j to its nearest neighbor within `selection`, or +inf."""
    nearest = np.float32(np.inf)
    for k in selection:
        if k != j:
            nearest = min(nearest, get_distance(store, j, k))
    return nearest


@numba.njit(REMOVE_SIGNATURE, cache=True)
def remove_trial(
    sep: NDArray[np.float32],
    store: DistanceStore,
    i_removed: np.int32,
    new_selection: NDArray[np.int32],
) -> None:
    """Update the separations of the items in `new_selection` after removing i_removed; leave the rest stale."""
    for j in new_selection:
        if get_distance(store, i_removed, j) <= sep[j]:
            sep[j] = _nearest_selected(store, j, new_selection)
