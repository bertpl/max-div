"""The metric layer defines which distances exist and how each one is computed for a pair.

The layer everything else in `_distance` is built on — the builds and the on-demand reads both go
through the same pair functions, which is what keeps stored and computed values bit-equal.
`_preprocess` is the one place a metric's vectors are preprocessed into the form its pair function expects.
"""

from ._distance_metric import NO_AXIS, NO_P, DistanceMetric
from ._pair import (
    _l2sq_pair,
    _metric_pair,
)
from ._preprocess import (
    preprocess_cosine_distance_vectors,
    preprocess_vectors,
    validate_cosine_distance_vectors,
    validate_vector_array_layout,
)

__all__ = [
    "NO_AXIS",
    "NO_P",
    "DistanceMetric",
    "_l2sq_pair",
    "_metric_pair",
    "preprocess_cosine_distance_vectors",
    "preprocess_vectors",
    "validate_cosine_distance_vectors",
    "validate_vector_array_layout",
]
