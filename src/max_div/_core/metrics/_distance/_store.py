"""Pluggable pairwise-distance storage: the read-only bundle njit kernels read distances from.

The bundle is a namedtuple of numpy arrays and scalars, so it can cross the njit boundary without
object-mode; fields a backend does not use hold zero-length arrays.  All solver kernels read
through `get_distance`, which keeps the storage layout out of every call site.
"""

from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

# =================================================================================================
#  DistanceStore
# =================================================================================================
# Backend selector values for DistanceStore.kind.
KIND_CONDENSED = np.int32(0)

# shared placeholders for the fields a backend does not use, so empty stores cost nothing
_EMPTY_1D = np.empty(0, dtype=np.float32)
_EMPTY_2D = np.empty((0, 0), dtype=np.float32)


class DistanceStore(NamedTuple):
    """Read-only pairwise-distance storage for n items, passable into njit kernels.

    Which field holds the distances is determined by `kind`; unused fields are zero-length
    arrays.  Instances are immutable by construction — kernels can only read, and copies of
    consuming objects can safely share one store.  Create instances via the factory methods,
    one per backend.
    """

    kind: np.int32
    n: np.int32
    pdist: NDArray[np.float32]  # (n*(n-1)/2,) condensed distances (scipy layout), KIND_CONDENSED
    matrix: NDArray[np.float32]  # placeholder for a full (n, n) distance matrix backend
    vectors: NDArray[np.float32]  # placeholder for an on-demand (n, d) vector backend
    metric_kind: np.int32  # pair-kernel selector for the on-demand backend

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def condensed(cls, pdist: NDArray[np.float32], n: int) -> "DistanceStore":
        """Return a DistanceStore reading from a condensed distance vector (scipy layout).

        :param pdist: ((n*(n-1))//2 ndarray) condensed pairwise distances, float32 C-contiguous.
        :param n: (int) number of items.
        """
        return cls(
            kind=KIND_CONDENSED,
            n=np.int32(n),
            pdist=pdist,
            matrix=_EMPTY_2D,
            vectors=_EMPTY_2D,
            metric_kind=np.int32(0),
        )


# The numba type of every DistanceStore instance (all stores share it: field dtypes are fixed and
# selectors are runtime values).  Signature strings cannot spell a namedtuple type, so kernels
# taking a store use signature *objects* built from this constant.
DISTANCE_STORE_TYPE = numba.typeof(DistanceStore.condensed(_EMPTY_1D, 0))


# =================================================================================================
#  Distance reads
# =================================================================================================
@numba.njit("int64(int32, int32, int32)", inline="always", cache=True)
def _condensed_index(i_lo: np.int32, i_hi: np.int32, n: np.int32) -> np.int64:
    """Return the condensed-vector offset of the (i_lo, i_hi) distance, with i_lo < i_hi, for n items.

    The offset is evaluated in int64: the intermediate ``n * i_lo`` grows like n² and overflows int32
    for n above ~46k, so the operands are widened before the multiply even though the final offset fits.
    """
    i_lo64 = np.int64(i_lo)
    return (np.int64(n) * i_lo64) + np.int64(i_hi) - ((i_lo64 + 2) * (i_lo64 + 1)) // 2


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int32, numba.int32), inline="always", cache=True)
def get_distance(store: DistanceStore, i: np.int32, j: np.int32) -> np.float32:
    """Return the distance between items i and j from whichever backend the store holds.

    Access-pattern note for loops over many pairs: fix one index and sweep the other in ascending
    order — stored backends then read (mostly) contiguous memory, where sweeping the *fixed* index
    instead strides ~n elements per step.  The tracker kernels all follow this fix-i-sweep-j shape.
    """
    if i == j:
        return np.float32(0.0)
    if i < j:
        return store.pdist[_condensed_index(i, j, store.n)]
    return store.pdist[_condensed_index(j, i, store.n)]
