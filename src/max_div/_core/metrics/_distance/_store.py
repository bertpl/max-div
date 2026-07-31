"""Pluggable pairwise-distance storage: the read-only bundle njit kernels read distances from.

The bundle is a namedtuple of numpy arrays and scalars, so it can cross the njit boundary without
object-mode; fields a backend does not use hold zero-length arrays.  All solver kernels read
through `get_distance`, which keeps the storage layout out of every call site.
"""

from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from ._compute import _l1_pair, _l2sq_pair, normalize_rows, validate_cosine_vectors
from ._enum import DistanceMetric

# =================================================================================================
#  DistanceStore
# =================================================================================================
# Backend selector values for DistanceStore.kind.
KIND_CONDENSED = np.int32(0)
KIND_LAZY = np.int32(1)

# Pair-kernel selector values for DistanceStore.metric_kind (lazy backend only).  Cosine holds
# pre-normalized vectors, so its pair read is the half-squared-L2 form on those.
_METRIC_KIND_L1 = np.int32(0)
_METRIC_KIND_L2 = np.int32(1)
_METRIC_KIND_L2S = np.int32(2)
_METRIC_KIND_COS = np.int32(3)

_METRIC_KINDS = {
    DistanceMetric.L1_MANHATTAN: _METRIC_KIND_L1,
    DistanceMetric.L2_EUCLIDEAN: _METRIC_KIND_L2,
    DistanceMetric.L2S_EUCLIDEAN_SQUARED: _METRIC_KIND_L2S,
    DistanceMetric.COSINE: _METRIC_KIND_COS,
}

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
    vectors: NDArray[np.float32]  # (n, d) vectors distances are computed from, KIND_LAZY
    metric_kind: np.int32  # pair-kernel selector, KIND_LAZY only

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

    @classmethod
    def lazy(cls, vectors: NDArray[np.float32], metric: DistanceMetric) -> "DistanceStore":
        """Return a DistanceStore computing distances on demand from the given vectors.

        Cosine holds the rows pre-normalized (the exact normalization the pairwise kernels use), so
        the on-demand pair read reuses the squared-L2 accumulation.

        :param vectors: (n x d ndarray) the vectors to compute distances from.
        :param metric: (DistanceMetric) the distance metric to use.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if metric == DistanceMetric.COSINE:
            validate_cosine_vectors(vectors)
            vectors = normalize_rows(vectors)
        return cls(
            kind=KIND_LAZY,
            n=np.int32(vectors.shape[0]),
            pdist=_EMPTY_1D,
            matrix=_EMPTY_2D,
            vectors=vectors,
            metric_kind=_METRIC_KINDS[metric],
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


@numba.njit(numba.float32(numba.float32[:, ::1], numba.int32, numba.int32, numba.int32), inline="always", cache=True)
def _lazy_pair(vectors: NDArray[np.float32], metric_kind: np.int32, i: np.int32, j: np.int32) -> np.float32:
    """Compute the distance between vectors i and j on demand, per the given pair-kernel selector.

    Each branch narrows exactly as the corresponding precomputing kernel does, so on-demand values
    are bit-equal to stored ones.
    """
    if metric_kind == _METRIC_KIND_L1:
        return np.float32(_l1_pair(vectors, i, j))
    if metric_kind == _METRIC_KIND_L2:
        return np.float32(np.sqrt(_l2sq_pair(vectors, i, j)))
    if metric_kind == _METRIC_KIND_L2S:
        return np.float32(_l2sq_pair(vectors, i, j))
    return np.float32(0.5 * _l2sq_pair(vectors, i, j))  # cosine: rows are pre-normalized


@numba.njit(numba.float32(DISTANCE_STORE_TYPE, numba.int32, numba.int32), inline="always", cache=True)
def get_distance(store: DistanceStore, i: np.int32, j: np.int32) -> np.float32:
    """Return the distance between items i and j from whichever backend the store holds.

    Access-pattern note for loops over many pairs: keep `i` fixed and sweep `j` in ascending
    order (the shape all tracker kernels follow).  Stored backends then read (mostly) contiguous
    memory; the swapped nesting — sweeping `i` under a fixed `j` — strides ~n elements per read.
    """
    if i == j:
        return np.float32(0.0)
    if store.kind == KIND_LAZY:
        return _lazy_pair(store.vectors, store.metric_kind, i, j)
    if i < j:
        return store.pdist[_condensed_index(i, j, store.n)]
    return store.pdist[_condensed_index(j, i, store.n)]
