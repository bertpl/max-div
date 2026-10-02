"""Distance storage decides how a solve stores its pairwise distances, and builds the distance stores.

- `storage` holds the user's choice of storage type and the per-store types that a solve resolves to.
- `memory_budget` sizes a full matrix in bytes, probes the machine's RAM, and refuses a matrix that cannot fit.
- `allocation` decides where the arrays of a distance store are placed in memory.
- `shared_memory` lets a worker process read a distance store that another process built in shared memory.
- `factory_base` holds what the distance store factory of every problem flavor shares.
- `factory_vector_problem` builds the stores of a vector problem, from its vectors.
- `factory_distance_problem` builds the store of a distance-input problem, from its given distances.

A problem class returns the factory for its own flavor, so this package does not depend on the
problem classes.
"""

from .factory_base import DistanceStoreFactory
from .factory_distance_problem import DistanceProblemDistanceStoreFactory
from .factory_vector_problem import VectorProblemDistanceStoreFactory
from .memory_budget import total_physical_memory_bytes
from .shared_memory import SharedStoreSpec, attached_distance_store
from .storage import DistanceStorageType, DistanceStorageTypes

__all__ = [
    "DistanceProblemDistanceStoreFactory",
    "DistanceStorageType",
    "DistanceStorageTypes",
    "DistanceStoreFactory",
    "SharedStoreSpec",
    "VectorProblemDistanceStoreFactory",
    "attached_distance_store",
    "total_physical_memory_bytes",
]
