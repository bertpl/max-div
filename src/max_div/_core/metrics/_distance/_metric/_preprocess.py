"""Some metrics read a preprocessed form of the vectors; this module is the only place that preprocessing happens.

`DistanceMetric.preprocesses_vectors` declares whether a metric needs it; `preprocess_vectors` applies
it.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from ._distance_metric import METRIC_KIND_COS, DistanceMetric


# =================================================================================================
#  Preconditions
# =================================================================================================
def validate_vector_layout(vectors: NDArray[np.float32]) -> None:
    """Raise ValueError unless `vectors` is a 2D float32 C-contiguous array, the form every distance read expects.

    Converting here would return a copy, but `preprocess_vectors` must return the input itself for a
    metric that does not preprocess; the problem constructor produces this form, so a violation is a
    caller bug.
    """
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not vectors.flags.c_contiguous:
        raise ValueError(
            f"vectors must be a 2D float32 C-contiguous array; got {vectors.ndim}D {vectors.dtype} "
            f"with C-contiguous={vectors.flags.c_contiguous}."
        )


# =================================================================================================
#  Preprocessing
# =================================================================================================
def preprocess_vectors(vectors: NDArray[np.float32], metric: DistanceMetric) -> NDArray[np.float32]:
    """Return the array the given metric's distance reads expect, following `metric.preprocesses_vectors`.

    The result is the input itself when the metric does not preprocess, and a new array when it
    does; the input is never written.

    Raises:
        ValueError: If `vectors` is not in the required form, or a cosine row is all-zero.
    """
    validate_vector_layout(vectors)
    if not metric.preprocesses_vectors:
        return vectors
    if metric.kind == METRIC_KIND_COS:
        validate_cosine_vectors(vectors)
        preprocessed = normalize_rows(vectors)
    else:  # pragma: no cover -- every preprocessing kind has a branch above; a kind that lacks one lands here
        raise NotImplementedError(
            f"{metric!r} declares that it preprocesses vectors, but no preprocessing exists for it."
        )
    if np.shares_memory(preprocessed, vectors):  # pragma: no cover
        raise RuntimeError(f"Preprocessing for {metric!r} returned an array sharing memory with its input.")
    return preprocessed


def validate_cosine_vectors(vectors: NDArray[np.float32]) -> None:
    """Raise ValueError if any vector is all-zero — cosine distance is undefined for zero vectors."""
    zero_rows = np.flatnonzero(~vectors.any(axis=1))
    if zero_rows.size > 0:
        raise ValueError(
            f"Cosine distance is undefined for zero vectors; found an all-zero vector at row {zero_rows[0]}."
        )


@numba.njit(numba.float32[:, ::1](numba.types.Array(numba.float32, 2, "C", readonly=True)), cache=True)
def normalize_rows(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a fresh float32 array with each row of `vectors` scaled to unit L2 norm.

    Norms accumulate in float64 and each element narrows to float32 on store, so the result is the
    exact normalization the cosine pair functions operate on.  Rows must not be all-zero
    (`validate_cosine_vectors` guards the public entry points).
    """
    n = vectors.shape[0]
    d = vectors.shape[1]
    normalized = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        acc = np.float64(0.0)
        for c in range(d):
            acc += np.float64(vectors[i, c]) * np.float64(vectors[i, c])
        norm = np.sqrt(acc)
        for c in range(d):
            normalized[i, c] = np.float32(np.float64(vectors[i, c]) / norm)
    return normalized
