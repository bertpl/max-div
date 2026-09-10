"""The solver's distance storage is decided here: the user's choice, the policy, and the store builds.

- `storage` holds the public choice.
- `memory_budget` sizes a full matrix in bytes, probes the machine's RAM, and refuses a matrix that cannot fit.
- `shared_memory` publishes a store's array for other processes, which attach to it by name.
- `build` resolves the choice and builds the store, in process or into shared memory.
"""

from .build import build_distance_store, build_shared_distance_store, select_distance_storage
from .memory_budget import total_physical_memory_bytes
from .shared_memory import SharedDistanceStore, SharedStoreSpec, attached_distance_store, publish_distance_store
from .storage import DistanceStorageType

__all__ = [
    "DistanceStorageType",
    "SharedDistanceStore",
    "SharedStoreSpec",
    "attached_distance_store",
    "build_distance_store",
    "build_shared_distance_store",
    "publish_distance_store",
    "select_distance_storage",
    "total_physical_memory_bytes",
]
