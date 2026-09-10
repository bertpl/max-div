"""The solver's distance storage is decided here: the user's choice, the policy, and the store factory.

- `storage` holds the public choice.
- `memory_budget` sizes a full matrix in bytes, probes the machine's RAM, and refuses a matrix that cannot fit.
- `allocation` decides where the arrays of a distance store are placed in memory.
- `shared_memory` lets a worker process read a distance store that another process built in shared memory.
- `factory` is the one place a problem and its distances become stores.
"""

from .factory import DistanceStoreFactory, StoreDistance
from .memory_budget import total_physical_memory_bytes
from .shared_memory import SharedStoreSpec, attached_distance_store
from .storage import DistanceStorageType

__all__ = [
    "DistanceStorageType",
    "DistanceStoreFactory",
    "SharedStoreSpec",
    "StoreDistance",
    "attached_distance_store",
    "total_physical_memory_bytes",
]
