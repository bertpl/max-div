"""Pairwise distances: what they mean, how they are computed, and where they are kept.

Four layers, each depending only on the ones before it.  `_metric` defines the distances themselves,
`_build` turns vectors into a distance matrix, `_store` holds that data and reads it back, and
`_shared_memory` puts a store where several processes can read one copy of it.
"""

from ._build import compute_full_matrix, expand_condensed
from ._metric import DistanceMetric, preprocess_vectors, validate_cosine_vectors
from ._shared_memory import SharedDistanceStore, SharedStoreSpec, attached_distance_store, publish_distance_store
from ._store import (
    DISTANCE_STORE_TYPE,
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceStore,
    get_distance,
    get_distance_full_matrix,
    get_distance_lazy,
)
