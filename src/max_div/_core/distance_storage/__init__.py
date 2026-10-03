"""The solver's distance storage is decided here: the user's choice, the policy, and the store factory.

- `storage` holds the public choice.
- `memory_budget` sizes a full matrix in bytes, probes the machine's RAM, and refuses a matrix that cannot fit.
- `data_matrix_source` says how each data matrix of a solve, an array that a distance store reads, is produced.
- `allocation` decides where each data matrix is placed in memory.
- `data_matrix_registry` produces the data matrices of a solve and returns each by its matrix id.
- `shared_memory` lets a worker process read the data matrices that another process published in shared memory.
- `factory` is the one place a problem and its distances become stores.
"""

from .factory import DistanceStoreFactory, stores_by_distance
from .memory_budget import total_physical_memory_bytes
from .shared_memory import PublishedDistanceStores
from .storage import DistanceStorageType, DistanceStorageTypes

__all__ = [
    "DistanceStorageType",
    "DistanceStorageTypes",
    "DistanceStoreFactory",
    "PublishedDistanceStores",
    "stores_by_distance",
    "total_physical_memory_bytes",
]
