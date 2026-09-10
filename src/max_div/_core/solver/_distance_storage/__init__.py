"""The solver's distance storage is decided here: the user's choice, the policy, and the store factory.

- `storage` holds the public choice.
- `memory_budget` sizes a full matrix in bytes, probes the machine's RAM, and refuses a matrix that cannot fit.
- `allocation` decides where a store's array lives.
- `shared_memory` publishes a store's array for other processes, which attach to it by name.
- `factory` is the one place a problem and its distances become stores.
"""

from .factory import DistanceStoreFactory, SharedStoreSet, StoreDistance
from .memory_budget import total_physical_memory_bytes
from .shared_memory import SharedDistanceStore, SharedStoreSpec, attached_distance_store, publish_distance_store
from .storage import DistanceStorageType

__all__ = [
    "DistanceStorageType",
    "DistanceStoreFactory",
    "SharedDistanceStore",
    "SharedStoreSet",
    "SharedStoreSpec",
    "StoreDistance",
    "attached_distance_store",
    "publish_distance_store",
    "total_physical_memory_bytes",
]
