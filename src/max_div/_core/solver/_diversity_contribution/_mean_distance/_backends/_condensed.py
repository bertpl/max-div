"""Mean-distance calculations for a condensed store.

One of three interchangeable modules — see this package's `__init__` for the pattern and why the backend is
chosen once per tracker rather than tested inside these loops.  Each module defines the same
three calculations over the same signatures, differing only in how a distance is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import get_distance_condensed

from .._signatures import ELEMENTS_SIGNATURE, UPDATE_SIGNATURE

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceStore


@numba.njit(ELEMENTS_SIGNATURE, cache=True)
def elements(out: NDArray[np.float32], store: DistanceStore, indices: NDArray[np.int32]) -> None:
    """Fill the given elements of `out` with each item's mean distance to all others."""
    den = np.float64(max(store.n - 1, 1))
    for idx in indices:
        row_sum = np.float64(0.0)
        for j in range(idx):
            row_sum += np.float64(get_distance_condensed(store, idx, j))
        for j in range(idx + 1, store.n):
            row_sum += np.float64(get_distance_condensed(store, idx, j))
        out[idx] = np.float32(row_sum / den)


@numba.njit(UPDATE_SIGNATURE, cache=True)
def add(dist_sums: NDArray[np.float64], store: DistanceStore, i_added: np.int32) -> None:
    """Update distance sums of each item wrt selection after adding i_added."""
    for j in range(i_added):
        dist_sums[j] += np.float64(get_distance_condensed(store, i_added, j))
    for j in range(i_added + 1, store.n):
        dist_sums[j] += np.float64(get_distance_condensed(store, i_added, j))


@numba.njit(UPDATE_SIGNATURE, cache=True)
def remove(dist_sums: NDArray[np.float64], store: DistanceStore, i_removed: np.int32) -> None:
    """Update distance sums of each item wrt selection after removing i_removed."""
    for j in range(i_removed):
        dist_sums[j] -= np.float64(get_distance_condensed(store, i_removed, j))
    for j in range(i_removed + 1, store.n):
        dist_sums[j] -= np.float64(get_distance_condensed(store, i_removed, j))
