"""Helpers shared by tests that need pairwise distances in scipy's condensed order."""

import numpy as np
from scipy.spatial.distance import squareform

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import compute_full_matrix


def marginals_and_joint_reference(a: np.ndarray, b: np.ndarray, joint_scale: float) -> float:
    """Return the marginals-and-joint distance of 2 vectors, computed in float64 with numpy."""
    gaps = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(min(gaps.min(), joint_scale * np.linalg.norm(gaps) ** len(gaps)))


def condensed_distances(vectors: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Return the pairwise distances in scipy's condensed order, taken from the full-matrix build."""
    return squareform(compute_full_matrix(vectors, metric), checks=False)
