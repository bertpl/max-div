"""A store holds a problem's pairwise distances read-only, in whichever layout it was built for.

A namedtuple of numpy arrays and scalars, so it crosses the njit boundary without object-mode;
fields a backend does not use hold zero-length arrays.  Which field carries the distances is what
`kind` selects, and `_reads` is the only place that knows how to index each one.

A lazy store holds the vectors as `preprocess_vectors` returns them for the metric it was built for,
so only metrics with the same preprocessing may share it.
"""

from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics._distance._build import compute_full_matrix, expand_condensed
from max_div._core.metrics._distance._metric import (
    NO_P,
    DistanceMetric,
    preprocess_vectors,
    validate_vector_array_layout,
)

# =================================================================================================
#  DistanceStore
# =================================================================================================
# Backend selector values for DistanceStore.kind.
KIND_LAZY = np.int32(0)
KIND_FULL_MATRIX = np.int32(1)

# One shared placeholder fills the fields a backend does not use, so empty stores cost nothing.  It is
# read-only because every store of a given backend hands out the same object, and because it makes every
# field of DISTANCE_STORE_TYPE read-only.
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
    matrix: NDArray[np.float32]  # (n, n) full distance matrix (exactly symmetric), KIND_FULL_MATRIX
    preprocessed_vectors: NDArray[np.float32]  # (n, d) the vectors as preprocessed for the metric, KIND_LAZY
    metric_kind: np.int32  # pair-function selector, KIND_LAZY only
    metric_p: np.float64  # `DistanceMetric.p`, KIND_LAZY only

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def lazy(cls, preprocessed_vectors: NDArray[np.float32], metric: DistanceMetric) -> "DistanceStore":
        """Return a DistanceStore computing distances on demand from vectors already preprocessed for the metric.

        The store adopts the array as a read-only view, without copying, so a store over a shared
        segment reads the segment's own bytes.  Preprocessing the user's vectors for the metric is the
        caller's job (`lazy_from_vectors` preprocesses, then calls `lazy`).

        Args:
            preprocessed_vectors: (n x d ndarray) the vectors as `preprocess_vectors` returns them for the
                metric.
            metric: (DistanceMetric) the distance metric the store computes.
        """
        validate_vector_array_layout(preprocessed_vectors)
        return cls(
            kind=KIND_LAZY,
            n=np.int32(preprocessed_vectors.shape[0]),
            matrix=_EMPTY_2D,
            preprocessed_vectors=_readonly(preprocessed_vectors),
            metric_kind=np.int32(metric.kind),
            metric_p=np.float64(metric.p),
        )

    @classmethod
    def lazy_from_vectors(cls, vectors: NDArray[np.float32], metric: DistanceMetric) -> "DistanceStore":
        """Return a lazy DistanceStore over the user's vectors, preprocessed for the metric first."""
        return cls.lazy(preprocess_vectors(vectors, metric), metric)

    @classmethod
    def full_matrix(cls, matrix: NDArray[np.float32]) -> "DistanceStore":
        """Return a DistanceStore reading from a full (n, n) distance matrix.

        The matrix must be float32, C-contiguous, and exactly symmetric with a zero diagonal —
        kernels read whichever of (i, j)/(j, i) suits their access pattern, so the two halves must
        be bit-equal.  Construction paths that cannot guarantee this by construction must repair
        or validate before wrapping.

        Args:
            matrix: ((n, n) ndarray) full pairwise-distance matrix.
        """
        return cls(
            kind=KIND_FULL_MATRIX,
            n=np.int32(matrix.shape[0]),
            matrix=_readonly(matrix),
            preprocessed_vectors=_EMPTY_2D,
            metric_kind=np.int32(0),
            metric_p=np.float64(NO_P),
        )

    @classmethod
    def full_matrix_from_vectors(cls, vectors: NDArray[np.float32], metric: DistanceMetric) -> "DistanceStore":
        """Return a full-matrix DistanceStore computed from the user's vectors.

        Each pair is computed once through the same pair arithmetic the lazy reads use, and written
        to both halves — so values are bit-equal across backends and symmetry is structural.

        Args:
            vectors: (n x d ndarray) the user's vectors, in the form `validate_vector_array_layout` accepts.
            metric: (DistanceMetric) the distance metric to use.
        """
        return cls.full_matrix(compute_full_matrix(vectors, metric))

    @classmethod
    def full_matrix_from_condensed(cls, condensed: NDArray[np.float32], n: int) -> "DistanceStore":
        """Return a full-matrix DistanceStore expanded from a condensed distance vector (scipy layout).

        Each condensed value is written to both halves, so the matrix is exactly symmetric and
        bit-equal to the condensed source.

        Args:
            condensed: ((n*(n-1))//2 ndarray) condensed pairwise distances, float32 C-contiguous.
            n: (int) number of items.
        """
        return cls.full_matrix(expand_condensed(condensed, n))


# The numba type of every DistanceStore instance (all stores share it: field dtypes are fixed and
# selectors are runtime values).  Signature strings cannot spell a namedtuple type, so kernels
# taking a store use signature *objects* built from this constant.
#
# Its arrays are typed read-only so that stores reading shared memory type-check: numba converts a
# writable array to a read-only parameter but never the reverse, so a writable type here would
# reject them, which would force the views into a segment several processes read to be writable.
DISTANCE_STORE_TYPE = numba.typeof(DistanceStore.full_matrix(_EMPTY_2D))
