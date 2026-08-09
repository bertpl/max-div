"""Reading one distance out of a store, for every layout a store can hold.

The single owner of layout knowledge: how a pair maps onto a condensed offset, which half of a full
matrix to touch, and when a distance is computed rather than looked up.  Everything else treats a
store as opaque.
"""

import numba
import numpy as np

from max_div._core.metrics._distance._metric import _metric_pair

from ._bundle import DISTANCE_STORE_TYPE, KIND_FULL_MATRIX, KIND_LAZY, DistanceStore


@numba.njit("int64(int32, int32, int32)", inline="always", cache=True)
def _condensed_index(i_lo: np.int32, i_hi: np.int32, n: np.int32) -> np.int64:
    """Return the condensed-vector offset of the (i_lo, i_hi) distance, with i_lo < i_hi, for n items.

    The offset is evaluated in int64: the intermediate ``n * i_lo`` grows like n² and overflows int32
    for n above ~46k, so the operands are widened before the multiply even though the final offset fits.
    """
    i_lo64 = np.int64(i_lo)
    return (np.int64(n) * i_lo64) + np.int64(i_hi) - ((i_lo64 + 2) * (i_lo64 + 1)) // 2


# A specialized reader per storage layout, for the loops that read one item's distance to every
# other.  They exist as a performance optimization over the generic `get_distance`, which measured
# about fifteen times slower in those loops.
#
# Each requires two preconditions that `get_distance` does not.  Neither is checked, and breaking
# either is fatal rather than wrong:
#
#   - the store holds the layout the reader is named after.  A reader handed another layout reads
#     a zero-length array.  Settled once per tracker, by looking the reader up by store kind.
#   - i differs from j.  The condensed layout has no diagonal, so a self-pair lands outside its
#     array.  Callers satisfy this by looping over a range that skips their own index.
@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int64, numba.int64), inline="always", cache=True)
def get_distance_full_matrix(store: DistanceStore, i: int | np.integer, j: int | np.integer) -> np.float32:
    """Read the distance between two distinct items from a full-matrix store."""
    return store.matrix[i, j]


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int64, numba.int64), inline="always", cache=True)
def get_distance_condensed(store: DistanceStore, i: int | np.integer, j: int | np.integer) -> np.float32:
    """Read the distance between two distinct items from a condensed store."""
    if i < j:
        return store.pdist[_condensed_index(np.int32(i), np.int32(j), store.n)]
    return store.pdist[_condensed_index(np.int32(j), np.int32(i), store.n)]


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int64, numba.int64), inline="always", cache=True)
def get_distance_lazy(store: DistanceStore, i: int | np.integer, j: int | np.integer) -> np.float32:
    """Compute the distance between two items from a lazy store's vectors."""
    return _metric_pair(store.vectors, store.metric_kind, np.int32(i), np.int32(j))


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int32, numba.int32), inline="always", cache=True)
def get_distance(store: DistanceStore, i: np.int32, j: np.int32) -> np.float32:
    """Return the distance between items i and j from whichever backend the store holds.

    For reads of a single pair, and for any caller that may ask for a self-pair.  Loops reading
    many pairs use `get_stored_distance` / `get_computed_distance` instead, which drop the two
    branches that would keep such a loop scalar.

    Access-pattern note for loops over many pairs: keep `i` fixed and sweep `j` in ascending
    order (the shape all tracker kernels follow).  Stored backends then read (mostly) contiguous
    memory; the swapped nesting — sweeping `i` under a fixed `j` — strides ~n elements per read.
    """
    if i == j:
        return np.float32(0.0)
    if store.kind == KIND_FULL_MATRIX:
        return store.matrix[i, j]
    if store.kind == KIND_LAZY:
        return _metric_pair(store.vectors, store.metric_kind, i, j)
    if i < j:
        return store.pdist[_condensed_index(i, j, store.n)]
    return store.pdist[_condensed_index(j, i, store.n)]
