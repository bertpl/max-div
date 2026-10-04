"""This package stores a solve's pairwise distances, following the storage type that the user chose.

"Storage type" names a `DistanceStorageType` value throughout this package; "kind" is reserved for
`DistanceStore.kind`, the compiled selector that a distance store carries.

- `storage` holds the public choice.
- `memory_budget` sizes a data matrix in bytes and probes the machine's RAM.
- `data_matrix_producer` says how each data matrix of a solve is produced; a data matrix is an array that a
  distance store reads.
- `storage_plan` decides, before a solve starts, how each distance of a solve is stored and which data
  matrices its distance stores read, and refuses a solve whose data matrices cannot fit in memory.
- `allocation` decides where each data matrix is placed in memory, and records where a shared-memory segment
  holds one.
- `data_matrix_readers` returns a solve's data matrices by matrix id, in a single solve or in a worker of a
  parallel solve.
- `data_matrix_publisher` produces the data matrices of a parallel solve in shared memory, for its workers.
"""

from .allocation import PublishedDataMatrixRecords
from .data_matrix_producer import AdoptingDataMatrixProducer, ComputingDataMatrixProducer, DataMatrixProducer
from .data_matrix_publisher import SharedMemoryDataMatrixPublisher
from .data_matrix_readers import InProcessDataMatrixReader, SharedMemoryDataMatrixReader
from .memory_budget import total_physical_memory_bytes
from .storage import DistanceStorageType, DistanceStorageTypes
from .storage_plan import USER_MATRIX_ID, DistanceStoragePlan

__all__ = [
    "USER_MATRIX_ID",
    "AdoptingDataMatrixProducer",
    "ComputingDataMatrixProducer",
    "DataMatrixProducer",
    "DistanceStoragePlan",
    "DistanceStorageType",
    "DistanceStorageTypes",
    "InProcessDataMatrixReader",
    "PublishedDataMatrixRecords",
    "SharedMemoryDataMatrixPublisher",
    "SharedMemoryDataMatrixReader",
    "total_physical_memory_bytes",
]
