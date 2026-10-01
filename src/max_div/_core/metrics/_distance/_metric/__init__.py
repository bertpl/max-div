"""The metric layer defines which distances exist and how each one is computed for a pair.

The layer everything else in `_distance` is built on — the builds and the on-demand reads both go
through the same pair functions, which is what keeps stored and computed values bit-equal.
Each metric class preprocesses its own vectors into the form its pair function expects.
"""

from ._distance_metric import NO_PARAM, DistanceMetric
from ._pair import (
    _l2sq_pair,
    _metric_pair,
)
from ._preprocess import validate_vector_array_layout

__all__ = [
    "NO_PARAM",
    "DistanceMetric",
    "_l2sq_pair",
    "_metric_pair",
    "validate_vector_array_layout",
]
