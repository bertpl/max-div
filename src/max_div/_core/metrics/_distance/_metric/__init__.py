"""The metric layer defines which distances exist and how each one is computed for a pair.

The layer everything else in `_distance` is built on — the builds and the on-demand reads both go
through the same pair functions, which is what keeps stored and computed values bit-equal.
"""

from ._distance_metric import DistanceMetric
from ._pair import (
    _l2sq_pair,
    _metric_pair,
    normalize_rows,
    validate_cosine_vectors,
)

__all__ = [
    "DistanceMetric",
    "_l2sq_pair",
    "_metric_pair",
    "normalize_rows",
    "validate_cosine_vectors",
]
