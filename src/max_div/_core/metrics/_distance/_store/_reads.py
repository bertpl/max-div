"""Every read of a single distance goes through one of the functions here, whatever layout the store holds.

This module is the single owner of layout knowledge: it decides which half of a full matrix to touch,
and when a distance is computed, not looked up.  Everything else treats a store as opaque.
"""

import numba
import numpy as np

from max_div._core.metrics._distance._metric import _metric_pair

from ._bundle import DISTANCE_STORE_TYPE, KIND_FULL_MATRIX, DistanceStore


# A specialized reader per storage layout, for the loops that read one item's distance to every
# other.  They exist as a performance optimization over the generic `get_distance`, which measured
# about fifteen times slower in those loops.
#
# Each requires a precondition that `get_distance` does not: the store holds the layout the reader
# is named after.  A reader handed another layout reads a zero-length array, which is fatal, not
# wrong.  Settled once per tracker, by looking the reader up by store kind.
@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int64, numba.int64), inline="always", cache=True)
def get_distance_full_matrix(store: DistanceStore, i: int | np.integer, j: int | np.integer) -> np.float32:
    """Read the distance between two items from a full-matrix store."""
    return store.matrix[i, j]


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int64, numba.int64), inline="always", cache=True)
def get_distance_lazy(store: DistanceStore, i: int | np.integer, j: int | np.integer) -> np.float32:
    """Compute the distance between two items from a lazy store's preprocessed array."""
    return _metric_pair(store.preprocessed_vectors, store.metric_kind, store.metric_p, np.int32(i), np.int32(j))


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int32, numba.int32), inline="always", cache=True)
def get_distance(store: DistanceStore, i: np.int32, j: np.int32) -> np.float32:
    """Return the distance between items i and j from whichever backend the store holds.

    Use it for reads of a single pair.  Loops reading many pairs use the layout-specific readers above,
    which drop the branch that would keep such a loop scalar.

    Access-pattern note for loops over many pairs: keep `i` fixed and sweep `j` in ascending
    order (the shape all tracker loops follow).  The full-matrix backend then reads contiguous
    memory; the swapped nesting — sweeping `i` under a fixed `j` — strides n elements per read.
    """
    if i == j:
        return np.float32(0.0)
    if store.kind == KIND_FULL_MATRIX:
        return store.matrix[i, j]
    return _metric_pair(store.preprocessed_vectors, store.metric_kind, store.metric_p, i, j)
