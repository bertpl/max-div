"""Pairwise distances: what they mean, how they are computed, and where they are kept.

Four layers, each depending only on the ones before it.  `_metric` defines the distances themselves,
`_build` turns vectors into distance data, `_store` holds that data and reads it back, and
`_shared_memory` puts a store where several processes can read one copy of it.
"""

from ._build import compute_full_matrix, compute_pdist, expand_condensed
from ._metric import DistanceMetric, validate_cosine_vectors
from ._shared_memory import SharedDistanceStore, SharedStoreSpec, attached_distance_store, publish_distance_store
from ._store import (
    DISTANCE_STORE_TYPE,
    KIND_CONDENSED,
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceStore,
    get_distance,
    get_distance_condensed,
    get_distance_full_matrix,
    get_distance_lazy,
)
