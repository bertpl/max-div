"""Pairwise distances: what they mean, how they are computed, and where they are kept.

Each layer depends only on the ones before it:

- `_metric` defines the distances themselves.
- `_build` turns vectors into a distance matrix.
- `_store` holds that data and reads it back.
"""

from ._build import compute_full_matrix, expand_condensed
from ._metric import NO_AXIS, NO_P, DistanceMetric, preprocess_vectors, validate_cosine_distance_vectors
from ._store import (
    DISTANCE_STORE_TYPE,
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceStore,
    get_distance,
    get_distance_full_matrix,
    get_distance_lazy,
)
