"""Helpers shared by tests that need pairwise distances in scipy's condensed order."""

import numpy as np
from scipy.spatial.distance import squareform

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import compute_full_matrix


def l2_and_projections_reference(a: np.ndarray, b: np.ndarray, l2_scale: float, k: int | None = None) -> float:
    """Return the L2-and-projections distance of 2 vectors in float64 with numpy; with `k`, the form for k items."""
    gaps = np.abs(a.astype(np.float64) - b.astype(np.float64))
    l2 = np.linalg.norm(gaps)
    if k is None:
        return float(min(gaps.min(), l2_scale * l2 ** len(gaps)))
    else:
        return float(min(gaps.min(), l2_scale * (k ** (1.0 / len(gaps)) - 1.0) / (k - 1.0) * l2))


def condensed_distances(vectors: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Return the pairwise distances in scipy's condensed order, taken from the full-matrix build."""
    return squareform(compute_full_matrix(vectors, metric), checks=False)
