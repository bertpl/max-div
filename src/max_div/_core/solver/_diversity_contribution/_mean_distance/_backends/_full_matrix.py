"""Mean-distance calculations for a full-matrix store.

One of the interchangeable modules in this package — see this package's `__init__` for the pattern
and why the backend is chosen once per tracker rather than tested inside these loops.  Each module
defines the same two calculations over the same signatures, differing only in how a distance is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import get_distance_full_matrix

from .._signatures import UPDATE_SIGNATURE

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceStore


@numba.njit(UPDATE_SIGNATURE, cache=True)
def add(dist_sums: NDArray[np.float64], store: DistanceStore, i_added: np.int32) -> None:
    """Update distance sums of each item wrt selection after adding i_added."""
    for j in range(i_added):
        dist_sums[j] += np.float64(get_distance_full_matrix(store, i_added, j))
    for j in range(i_added + 1, store.n):
        dist_sums[j] += np.float64(get_distance_full_matrix(store, i_added, j))


@numba.njit(UPDATE_SIGNATURE, cache=True)
def remove(dist_sums: NDArray[np.float64], store: DistanceStore, i_removed: np.int32) -> None:
    """Update distance sums of each item wrt selection after removing i_removed."""
    for j in range(i_removed):
        dist_sums[j] -= np.float64(get_distance_full_matrix(store, i_removed, j))
    for j in range(i_removed + 1, store.n):
        dist_sums[j] -= np.float64(get_distance_full_matrix(store, i_removed, j))
