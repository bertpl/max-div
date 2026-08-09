"""Pluggable pairwise-distance storage: the read-only bundle njit kernels read distances from.

The bundle is a namedtuple of numpy arrays and scalars, so it can cross the njit boundary without
object-mode; fields a backend does not use hold zero-length arrays.  All solver kernels read
through `get_distance`, which keeps the storage layout out of every call site.
"""

from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from ._build import compute_full_matrix, expand_condensed
from ._compute import _METRIC_KINDS, _metric_pair, normalize_rows, validate_cosine_vectors
from ._enum import DistanceMetric

# =================================================================================================
#  DistanceStore
# =================================================================================================
# Backend selector values for DistanceStore.kind.
KIND_CONDENSED = np.int32(0)
KIND_LAZY = np.int32(1)
KIND_FULL_MATRIX = np.int32(2)

# Shared placeholders for the fields a backend does not use, so empty stores cost nothing.  Read-only
# because every store of a given backend hands out the same two objects; see DISTANCE_STORE_TYPE for
# what that buys beyond the obvious.
_EMPTY_1D = np.empty(0, dtype=np.float32)
_EMPTY_1D.flags.writeable = False
_EMPTY_2D = np.empty((0, 0), dtype=np.float32)
_EMPTY_2D.flags.writeable = False


def _readonly(array: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a view of `array` that cannot be written through, sharing its memory."""
    view = array.view()
    view.flags.writeable = False
    return view


class DistanceStore(NamedTuple):
    """Read-only pairwise-distance storage for n items, passable into njit kernels.

    Which field holds the distances is determined by `kind`; unused fields are zero-length
    arrays.  Instances are immutable by construction — kernels can only read, and copies of
    consuming objects can safely share one store.  Create instances via the factory methods,
    one per backend.

    The factories hold their array as a read-only *view* of what they were given, so a store still
    shares memory with its source (no copy is made) while nothing can write through the store
    itself.  That matters most when the source is a shared-memory segment several processes read,
    where a stray write corrupts every reader rather than failing where it happened.
    """

    kind: np.int32
    n: np.int32
    pdist: NDArray[np.float32]  # (n*(n-1)/2,) condensed distances (scipy layout), KIND_CONDENSED
    matrix: NDArray[np.float32]  # (n, n) full distance matrix (exactly symmetric), KIND_FULL_MATRIX
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
            pdist=_readonly(pdist),
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
            vectors=_readonly(vectors),
            metric_kind=_METRIC_KINDS[metric],
        )

    @classmethod
    def lazy_prepared(cls, vectors: NDArray[np.float32], metric_kind: np.int32) -> "DistanceStore":
        """Return a lazy DistanceStore over vectors already in the form the pair kernels expect.

        `lazy` prepares its input — for cosine, normalizing the rows into a fresh array — and so
        cannot be used by a caller that must keep reading the exact array it was handed.  Attaching
        to a published store is that caller: normalizing there would replace the shared buffer with
        a private copy, silently undoing the sharing.

        :param vectors: (n x d ndarray) vectors in final form, float32 C-contiguous.
        :param metric_kind: (int32) pair-kernel selector, as `lazy` would have derived from the metric.
        """
        return cls(
            kind=KIND_LAZY,
            n=np.int32(vectors.shape[0]),
            pdist=_EMPTY_1D,
            matrix=_EMPTY_2D,
            vectors=_readonly(vectors),
            metric_kind=metric_kind,
        )

    @classmethod
    def full_matrix(cls, matrix: NDArray[np.float32]) -> "DistanceStore":
        """Return a DistanceStore reading from a full (n, n) distance matrix.

        The matrix must be float32, C-contiguous, and exactly symmetric with a zero diagonal —
        kernels read whichever of (i, j)/(j, i) suits their access pattern, so the two halves must
        be bit-equal.  Construction paths that cannot guarantee this by construction must repair
        or validate before wrapping.

        :param matrix: ((n, n) ndarray) full pairwise-distance matrix.
        """
        return cls(
            kind=KIND_FULL_MATRIX,
            n=np.int32(matrix.shape[0]),
            pdist=_EMPTY_1D,
            matrix=_readonly(matrix),
            vectors=_EMPTY_2D,
            metric_kind=np.int32(0),
        )

    @classmethod
    def full_matrix_from_vectors(cls, vectors: NDArray[np.float32], metric: DistanceMetric) -> "DistanceStore":
        """Return a full-matrix DistanceStore computed from vectors, exactly symmetric by construction.

        Each pair is computed once through the same pair kernels the condensed and lazy paths use,
        and written to both halves — so values are bit-equal across backends and symmetry is
        structural.

        :param vectors: (n x d ndarray) the vectors to compute distances from.
        :param metric: (DistanceMetric) the distance metric to use.
        """
        return cls.full_matrix(compute_full_matrix(vectors, metric))

    @classmethod
    def full_matrix_from_condensed(cls, pdist: NDArray[np.float32], n: int) -> "DistanceStore":
        """Return a full-matrix DistanceStore expanded from a condensed distance vector (scipy layout).

        Each condensed value is written to both halves, so the matrix is exactly symmetric and
        bit-equal to the condensed source.

        :param pdist: ((n*(n-1))//2 ndarray) condensed pairwise distances, float32 C-contiguous.
        :param n: (int) number of items.
        """
        return cls.full_matrix(expand_condensed(pdist, n))


# The numba type of every DistanceStore instance (all stores share it: field dtypes are fixed and
# selectors are runtime values).  Signature strings cannot spell a namedtuple type, so kernels
# taking a store use signature *objects* built from this constant.
#
# Its arrays are typed read-only, which numba treats as the wider type: a store holding writable
# arrays converts to it, while one holding read-only arrays does not convert the other way.  Typing
# it writable would therefore reject the read-only views a store attached from shared memory hands
# to the kernels — and rejecting them is what would force those views to be writable, in a segment
# several processes read at once.
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
