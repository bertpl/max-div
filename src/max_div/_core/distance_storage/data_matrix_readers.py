"""A distance store reads its data matrix through a `DataMatrixReader`, and a solve uses 1 of its 2 subclasses.

`DataMatrixReader`, in `max_div._core.metrics._distance`, is the interface: it returns a data
matrix by its matrix id, and a `DistanceSpec` builds its distance store from it.  The 2 subclasses
serve the 2 ways a solve runs:

- `InProcessDataMatrixReader` serves a single solve: it produces every data matrix in this process,
  from the producers of the distance storage plan, and returns them from there.
- `SharedMemoryDataMatrixReader` serves a worker of a parallel solve: it reads the data matrices that
  the parent process published in shared memory with `SharedMemoryDataMatrixPublisher`, and finds
  them through their `PublishedDataMatrixRecords`.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from max_div._core._utils import attach_shared_memory_segment
from max_div._core.metrics._distance import DataMatrixReader

from .allocation import InProcessDataMatrixAllocator, PublishedDataMatrixRecords
from .data_matrix_producer import DataMatrixProducer

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory


# ==================================================================================================
#  InProcessDataMatrixReader
# ==================================================================================================
class InProcessDataMatrixReader(DataMatrixReader):
    """An in-process data matrix reader produces a single solve's data matrices in this process and reads them by id.

    Each data matrix is produced once, so 2 distance stores that read the same matrix id read the
    same array.
    """

    def __init__(self, producers: Mapping[int, DataMatrixProducer]) -> None:
        """Produce every data matrix in this process.

        Args:
            producers: the producer of each data matrix, by matrix id.
        """
        self._arrays = DataMatrixProducer.produce_all(producers, InProcessDataMatrixAllocator())

    def array(self, matrix_id: int) -> NDArray[np.float32]:
        """Return the data matrix with the given id."""
        return self._arrays[matrix_id]


# ==================================================================================================
#  SharedMemoryDataMatrixReader
# ==================================================================================================
class SharedMemoryDataMatrixReader(DataMatrixReader):
    """A shared-memory data matrix reader reads the data matrices that another process published, by matrix id.

    Entering the `with` block attaches to every published segment; leaving closes this process's
    mapping of each segment and never unlinks a segment, which belongs to the process that published
    it.  Read the data matrices, and the distance stores over them, only inside the block.
    """

    def __init__(self, published_matrix_records: PublishedDataMatrixRecords) -> None:
        """Keep the records of the published data matrices; the segments are attached when the block is entered."""
        self._published_matrix_records = published_matrix_records
        self._segments: list[SharedMemory] = []
        self._arrays: dict[int, NDArray[np.float32]] = {}

    def __enter__(self) -> "SharedMemoryDataMatrixReader":
        """Attach to every published segment and return this reader.

        Raises:
            FileNotFoundError: If a segment no longer exists; the segments attached before it are closed.
        """
        try:
            for matrix_id, published_matrix_record in self._published_matrix_records.records.items():
                segment = attach_shared_memory_segment(published_matrix_record.segment_name)
                self._segments.append(segment)
                self._arrays[matrix_id] = published_matrix_record.array_over(segment)
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close this process's mapping of every attached segment."""
        self._close()

    def array(self, matrix_id: int) -> NDArray[np.float32]:
        """Return the data matrix with the given id, read from its segment."""
        return self._arrays[matrix_id]

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    def _close(self) -> None:
        """Close this process's mapping of every attached segment, and forget the arrays over them."""
        self._arrays.clear()
        for segment in self._segments:
            segment.close()
        self._segments.clear()
