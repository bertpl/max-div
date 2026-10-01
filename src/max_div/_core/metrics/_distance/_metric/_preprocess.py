"""Vector-array helpers behind `DistanceMetric.preprocess`: the layout every distance function expects, and row scaling.

Each metric class preprocesses its own vectors; this module holds what is shared, or what numba
must compile.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit


def validate_vector_array_layout(vectors: NDArray[np.float32]) -> None:
    """Raise ValueError unless `vectors` is a 2D float32 C-contiguous array, as every distance function expects."""
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not vectors.flags.c_contiguous:
        raise ValueError(
            f"vectors must be a 2D float32 C-contiguous array; got {vectors.ndim}D {vectors.dtype} "
            f"with C-contiguous={vectors.flags.c_contiguous}."
        )


# Serves the cosine metric class only, but stays module-level because numba compiles it.
@lazy_njit(numba.float32[:, ::1](numba.types.Array(numba.float32, 2, "C", readonly=True)), cache=True)
def _normalize_rows(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale each row to unit L2 norm into a fresh float32 array.

    Norms accumulate in float64 and each element narrows to float32 on store, so the result is the
    exact normalization the cosine pair functions operate on.  Rows must not be all-zero.
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
