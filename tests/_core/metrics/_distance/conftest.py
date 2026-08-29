"""Fixtures shared across the distance test tree live here."""

import pytest

from max_div._core.metrics import DistanceMetric

# NAMED_METRICS lists every metric the factory methods can produce, for tests that must hold for
# each of them.
NAMED_METRICS = (
    DistanceMetric.l1_manhattan(),
    DistanceMetric.l2_euclidean(),
    DistanceMetric.l2s_euclidean_squared(),
    DistanceMetric.linf_chebyshev(),
    DistanceMetric.cosine(),
)


@pytest.fixture(params=NAMED_METRICS, ids=repr)
def metric(request: pytest.FixtureRequest) -> DistanceMetric:
    """Return each metric from NAMED_METRICS in turn."""
    return request.param
