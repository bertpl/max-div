"""Separation calculations for a condensed store.

One of three interchangeable modules — see this package's `__init__` for the pattern and why the
layout is chosen once per tracker rather than tested inside these loops.  Each module defines
the same calculations over the same signatures, differing only in how a distance is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance_condensed

from .._signatures import ADD_MANY_SIGNATURE, ADD_SIGNATURE, ELEMENTS_SIGNATURE, REMOVE_SIGNATURE

if TYPE_CHECKING:
    from numpy.typing import NDArray


@numba.njit(ELEMENTS_SIGNATURE, cache=True)
def elements(sep: NDArray[np.float32], store: DistanceStore, indices: NDArray[np.int32]) -> None:
    """Fill the given elements of `sep` with each item's separation wrt all others."""
    for idx in indices:
        row_min = np.float32(np.inf)
        for j in range(idx):
            row_min = min(row_min, get_distance_condensed(store, idx, j))
        for j in range(idx + 1, store.n):
            row_min = min(row_min, get_distance_condensed(store, idx, j))
        sep[idx] = row_min


@numba.njit(ADD_SIGNATURE, cache=True)
def add(sep: NDArray[np.float32], store: DistanceStore, i_added: np.int32) -> None:
    """Update separation of each item wrt selection after adding i_added."""
    for j in range(i_added):
        sep[j] = min(sep[j], get_distance_condensed(store, i_added, j))
    for j in range(i_added + 1, store.n):
        sep[j] = min(sep[j], get_distance_condensed(store, i_added, j))


@numba.njit(ADD_MANY_SIGNATURE, cache=True)
def add_many(sep: NDArray[np.float32], store: DistanceStore, i_added_many: NDArray[np.int32]) -> None:
    """Update separation of each item wrt selection after adding all of i_added_many.

    One fused pass over all items instead of one `add` sweep per added item: the same distance
    pairs, so the result is identical, but `sep` is read and written once.
    """
    for j in range(store.n):
        nearest = sep[j]
        for i in i_added_many:
            if i != j:
                nearest = min(nearest, get_distance_condensed(store, i, j))
        sep[j] = nearest


@numba.njit(ADD_MANY_SIGNATURE, parallel=True, cache=True)
def add_many_parallel(sep: NDArray[np.float32], store: DistanceStore, i_added_many: NDArray[np.int32]) -> None:
    """Parallel `add_many`: items are independent, and `min` is order-free, so results are identical."""
    for j in numba.prange(store.n):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
        nearest = sep[j]
        for i in i_added_many:
            if i != j:
                nearest = min(nearest, get_distance_condensed(store, i, j))
        sep[j] = nearest


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int64, numba.int32[::1]), inline="always", cache=True)
def _nearest_selected(store: DistanceStore, j: int | np.integer, selection: NDArray[np.int32]) -> np.float32:
    """Return the distance from item j to its nearest neighbor within `selection`, or +inf."""
    nearest = np.float32(np.inf)
    for k in selection:
        if k != j:
            nearest = min(nearest, get_distance_condensed(store, j, k))
    return nearest


@numba.njit(REMOVE_SIGNATURE, cache=True)
def remove(
    sep: NDArray[np.float32],
    store: DistanceStore,
    i_removed: np.int32,
    new_selection: NDArray[np.int32],
) -> None:
    """Update separation of each item wrt selection after removing i_removed.

    An item whose nearest selected neighbor was the removed one has to find a new nearest, which
    is the rescan against `new_selection`; the rest keep the separation they had.
    """
    for j in range(i_removed):
        if get_distance_condensed(store, i_removed, j) <= sep[j]:
            sep[j] = _nearest_selected(store, j, new_selection)
    for j in range(i_removed + 1, store.n):
        if get_distance_condensed(store, i_removed, j) <= sep[j]:
            sep[j] = _nearest_selected(store, j, new_selection)
