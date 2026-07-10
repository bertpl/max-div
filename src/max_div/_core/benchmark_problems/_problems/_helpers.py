import numpy as np


def sort_vectors(vectors: np.ndarray) -> np.ndarray:
    """Sort vectors by increasing L2 row-norm to provide deterministic ordering.

    Args:
        vectors (np.ndarray): Array of shape (n, d) containing n vectors of dimension d.

    Returns:
        np.ndarray: Sorted array of vectors.
    """
    norms = np.linalg.norm(vectors, axis=1)
    order = np.argsort(norms, kind="mergesort")
    return vectors[order]
