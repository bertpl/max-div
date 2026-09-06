"""Fixtures shared across the distance test tree live here."""

import pytest

from max_div._core.metrics import DistanceMetric

# NAMED_METRICS lists every metric with a factory method of its own.
NAMED_METRICS = (
    DistanceMetric.l1_manhattan(),
    DistanceMetric.l2_euclidean(),
    DistanceMetric.l2s_euclidean_squared(),
    DistanceMetric.linf_chebyshev(),
    DistanceMetric.cosine(),
    DistanceMetric.geometric_mean(),
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
