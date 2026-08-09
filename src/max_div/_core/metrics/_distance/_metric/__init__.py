"""What a distance is: the metrics on offer, and the per-pair arithmetic behind each of them.

The layer everything else in `_distance` is built on — the builds and the on-demand reads both go
through the same pair functions, which is what keeps stored and computed values bit-equal.
"""

from ._enum import DistanceMetric
from ._pair import (
    _METRIC_KINDS,
    _l2sq_pair,
    _metric_pair,
    normalize_rows,
    validate_cosine_vectors,
)

__all__ = [
    "_METRIC_KINDS",
    "DistanceMetric",
    "_l2sq_pair",
    "_metric_pair",
    "normalize_rows",
    "validate_cosine_vectors",
]
