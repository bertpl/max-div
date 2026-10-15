"""These helpers compute or read pairwise distances for tests."""

import numba
import numpy as np
from scipy.spatial.distance import squareform

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore, compute_full_matrix, get_distance
from tests.helpers import as_python_callable


def l2_and_projections_reference(a: np.ndarray, b: np.ndarray, l2_scale: float, k: int | None = None) -> float:
    """Return the L2-and-projections distance of 2 vectors in float64 with numpy; with `k`, the form for k items."""
    gaps = np.abs(a.astype(np.float64) - b.astype(np.float64))
    l2 = np.linalg.norm(gaps)
    if k is None:
        return float(min(gaps.min(), l2_scale * l2 ** len(gaps)))
    else:
        return float(min(gaps.min(), l2_scale * (k ** (1.0 / len(gaps)) - 1.0) / (k - 1.0) * l2))


@numba.njit(fastmath={"reassoc", "contract"})
def custom_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Return the L1 distance of 2 vectors, written as a user would write it for `DistanceMetric.custom`.

    It accumulates in float64 in coordinate order under the fastmath flags of the built-in metrics, so it
    computes the same bits as `DistanceMetric.l1_manhattan()`.
    """
    acc = 0.0
    for c in range(a.shape[0]):
        acc += abs(np.float64(a[c]) - np.float64(b[c]))
    return acc


def all_pair_distances(store: DistanceStore) -> list[float]:
    """Return every (i, j) distance that the store reports, self-pairs included."""
    return [
        float(as_python_callable(get_distance)(store, np.int32(i), np.int32(j)))
        for i in range(store.n)
        for j in range(store.n)
    ]


def condensed_distances(vectors: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Return the pairwise distances in scipy's condensed order, taken from the full-matrix build."""
    return squareform(compute_full_matrix(vectors, metric), checks=False)
