"""Fixtures shared across the distance test tree live here."""

import numba
import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric


@pytest.fixture
def vectors() -> np.ndarray:
    """Return a small float32 C-contiguous array with no zero rows, so every metric accepts it."""
    return np.ascontiguousarray(np.random.default_rng(7).random((6, 3), dtype=np.float32) + 0.1)


@numba.njit(fastmath={"reassoc", "contract"})
def custom_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Return the L1 distance of 2 vectors, written as a user would write it for `DistanceMetric.custom`.

    It accumulates in float64 in coordinate order under the fastmath flags of the built-in metrics, so it
    computes the bits that `DistanceMetric.l1_manhattan()` computes.
    """
    acc = 0.0
    for c in range(a.shape[0]):
        acc += abs(np.float64(a[c]) - np.float64(b[c]))
    return acc


# NAMED_METRICS lists every metric with a factory method of its own; the along-axis one reads a
# coordinate that every test array has.
NAMED_METRICS = (
    DistanceMetric.l1_manhattan(),
    DistanceMetric.l2_euclidean(),
    DistanceMetric.l2s_euclidean_squared(),
    DistanceMetric.linf_chebyshev(),
    DistanceMetric.cosine(),
    DistanceMetric.geometric_mean(),
    DistanceMetric.l_minus_inf(),
    DistanceMetric.along_axis(1),
    DistanceMetric.l2_and_projections(),
    DistanceMetric.l2_and_projections(l2_scale=0.25),
    DistanceMetric.l2_and_projections(k=100),
    DistanceMetric.custom(custom_l1),
)

# MINKOWSKI_METRICS covers each Minkowski kind once: generic and specialized, rooted and not.
MINKOWSKI_METRICS = (
    DistanceMetric.minkowski(3),
    DistanceMetric.minkowski(3, root=False),
    DistanceMetric.minkowski(0.5),
    DistanceMetric.minkowski(0.5, root=False),
    DistanceMetric.minkowski(0.25),
    DistanceMetric.minkowski(0.25, root=False),
    DistanceMetric.minkowski(0.125),
    DistanceMetric.minkowski(0.125, root=False),
)


@pytest.fixture(params=NAMED_METRICS + MINKOWSKI_METRICS, ids=repr)
def metric(request: pytest.FixtureRequest) -> DistanceMetric:
    """Return each metric from NAMED_METRICS and MINKOWSKI_METRICS in turn."""
    return request.param
