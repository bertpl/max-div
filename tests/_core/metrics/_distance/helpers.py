"""Helpers shared by tests that need pairwise distances in scipy's condensed order."""

import numpy as np
from scipy.spatial.distance import squareform

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import compute_full_matrix


def condensed_distances(vectors: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Return the pairwise distances in scipy's condensed order, taken from the full-matrix build."""
    return squareform(compute_full_matrix(vectors, metric), checks=False)
