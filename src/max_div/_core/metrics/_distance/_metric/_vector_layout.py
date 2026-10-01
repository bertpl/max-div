"""This module holds the vector-array layout check that the metric classes and the distance store share."""

import numpy as np
from numpy.typing import NDArray


def validate_vector_array_layout(vectors: NDArray[np.float32]) -> None:
    """Raise ValueError unless `vectors` is a 2D float32 C-contiguous array, as every distance function expects."""
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not vectors.flags.c_contiguous:
        raise ValueError(
            f"vectors must be a 2D float32 C-contiguous array; got {vectors.ndim}D {vectors.dtype} "
            f"with C-contiguous={vectors.flags.c_contiguous}."
        )
