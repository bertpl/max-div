import numpy as np


def make_banded_vectors_and_bands(n: int, m: int) -> tuple[np.ndarray, list[list[int]]]:
    """Generate the shared 2D vector cloud and dimension-0 band partition used by C1 and C2.

    Vectors are uniform x gaussian (semi-uniform density), sorted by L2 row-norm.  Dimension 0 is
    split into m equal-width bands over [0, 1]; every index lands in exactly one band, so the bands
    partition the population.

    Returns:
        `(vectors, bands)` where `bands[i]` lists the vector indices whose dimension-0 value falls
        in band i.
    """
    np.random.seed(42)
    uniform_col = np.random.rand(n, 1)
    gaussian_col = np.random.randn(n, 1)
    vectors = np.concatenate((uniform_col, gaussian_col), axis=1).astype(np.float32)
    vectors = sort_vectors(vectors)

    bands: list[list[int]] = [[] for _ in range(m)]
    for idx in range(n):
        # values are uniform in [0,1); min() guards the (measure-zero) value 1.0 edge
        band = min(int(vectors[idx, 0] * m), m - 1)
        bands[band].append(idx)
    return vectors, bands


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
