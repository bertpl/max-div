"""Pairwise distances: what they mean, how they are computed, and where they are kept.

Each layer depends only on the ones before it:

- `_metric` defines the distances themselves.
- `_build` turns vectors into a distance matrix.
- `_store` holds that data and reads it back.
- `_spec` says which data matrix a set of distances is read from, and builds the store that reads it.
"""

from ._build import compute_full_matrix, expand_condensed
from ._metric import NO_FLOAT_PARAM, DistanceMetric
from ._spec import DataMatrixReader, DistanceSpec, PrecomputedDistanceSpec, VectorDistanceSpec
from ._store import (
    DISTANCE_STORE_TYPE,
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceStore,
    get_distance,
    get_distance_full_matrix,
    get_distance_lazy,
)
