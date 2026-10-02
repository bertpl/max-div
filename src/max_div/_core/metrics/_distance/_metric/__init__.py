"""The metric layer defines which distances exist and how each one is computed for a pair.

The layer everything else in `_distance` is built on — the builds and the on-demand reads both go
through the same pairwise distance functions, which is what keeps stored and computed values bit-equal.
Each metric class preprocesses its own vectors into the form that its pairwise distance function expects.
"""

from ._distance_metric import NO_FLOAT_PARAM, DistanceMetric
from ._pairwise_distance import (
    _l2sq_distance,
    _pairwise_distance,
)
from ._vector_layout import validate_vector_array_layout

__all__ = [
    "NO_FLOAT_PARAM",
    "DistanceMetric",
    "_l2sq_distance",
    "_pairwise_distance",
    "validate_vector_array_layout",
]
