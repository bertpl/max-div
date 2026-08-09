"""Holding a problem's pairwise distances, and reading one back out.

`_bundle` owns what a store *is* — the layouts, and the factories that build one over each; `_reads`
owns how a distance is fetched from whichever layout a store holds.
"""

from ._bundle import (
    DISTANCE_STORE_TYPE,
    KIND_CONDENSED,
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceStore,
)
from ._reads import (
    get_distance,
    get_distance_condensed,
    get_distance_full_matrix,
    get_distance_lazy,
)

__all__ = [
    "DISTANCE_STORE_TYPE",
    "KIND_CONDENSED",
    "KIND_FULL_MATRIX",
    "KIND_LAZY",
    "DistanceStore",
    "get_distance",
    "get_distance_condensed",
    "get_distance_full_matrix",
    "get_distance_lazy",
]
