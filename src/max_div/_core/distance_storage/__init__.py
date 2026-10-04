"""This package stores a solve's pairwise distances, following the storage type that the user chose.

- `storage` holds the public choice.
- `memory_budget` sizes a full matrix in bytes, probes the machine's RAM, and refuses a matrix that cannot fit.
- `data_matrix_producer` says how each data matrix of a solve is produced; a data matrix is an array that a
  distance store reads.
- `allocation` decides where each data matrix is placed in memory, and records where a shared-memory segment
  holds one.
- `data_matrix_readers` returns a solve's data matrices by matrix id, in a single solve or in a worker of a
  parallel solve.
- `data_matrix_publisher` produces the data matrices of a parallel solve in shared memory, for its workers.
- `factory` is the one place a problem and its distances become stores.
"""

from .factory import DistanceStoreFactory, PublishedDistanceStoresRecord, stores_by_distance
from .memory_budget import total_physical_memory_bytes
from .storage import DistanceStorageType, DistanceStorageTypes

__all__ = [
    "DistanceStorageType",
    "DistanceStorageTypes",
    "DistanceStoreFactory",
    "PublishedDistanceStoresRecord",
    "stores_by_distance",
    "total_physical_memory_bytes",
]
