"""The read-only bundle that holds a problem's pairwise distances, in whichever layout it uses.

A namedtuple of numpy arrays and scalars, so it crosses the njit boundary without object-mode;
fields a backend does not use hold zero-length arrays.  Which field carries the distances is what
`kind` selects, and `_reads` is the only place that knows how to index each one.
"""

from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics._distance._build import compute_full_matrix, expand_condensed
from max_div._core.metrics._distance._metric import (
    _METRIC_KINDS,
    DistanceMetric,
    normalize_rows,
    validate_cosine_vectors,
)

# =================================================================================================
#  DistanceStore
# =================================================================================================
# Backend selector values for DistanceStore.kind.
KIND_CONDENSED = np.int32(0)
KIND_LAZY = np.int32(1)
KIND_FULL_MATRIX = np.int32(2)

# Shared placeholders for the fields a backend does not use, so empty stores cost nothing.  Read-only
# because every store of a given backend hands out the same two objects, and because it makes every
# field of DISTANCE_STORE_TYPE read-only.
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

    The factories hold their array as a read-only *view* of what they were given: a store shares
    memory with its source, and nothing can write through the store itself.
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
        """Return a lazy DistanceStore over vectors already in the form the distance reads expect.

        `lazy` prepares its input — for cosine, normalizing the rows into a fresh array — so it
        cannot serve a caller that must keep reading the exact array it was handed, such as a store
        over a shared segment.

        :param vectors: (n x d ndarray) vectors in final form, float32 C-contiguous.
        :param metric_kind: (int32) metric selector, as `lazy` would have derived from the metric.
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
# Its arrays are typed read-only so that stores reading shared memory type-check: numba converts a
# writable array to a read-only parameter but never the reverse, so a writable type here would
# reject them, which would force the views into a segment several processes read to be writable.
DISTANCE_STORE_TYPE = numba.typeof(DistanceStore.condensed(_EMPTY_1D, 0))
