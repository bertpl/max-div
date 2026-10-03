"""An allocator decides where each data matrix of a solve is placed in memory.

A data matrix is an array that a distance store reads: a full distance matrix, or the vectors of
a lazy distance store.  Each data matrix of a solve has a matrix id.

A `DataMatrixSource` decides what a data matrix contains; the allocator decides only where it
lives: in this process, or in a shared-memory segment that worker processes can read.  There is
one allocator class per placement, and a data matrix source produces its data matrix the same way
whichever allocator it is given.

A data matrix source asks an allocator for one of 2 things:

- `allocate` returns an empty, writable buffer that the data matrix source then fills, for example a full
  distance matrix that is computed straight into its final place.
- `adopt` takes an array that already exists in its final form, for example the user's own vectors,
  and returns the array that a distance store will read from.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from max_div._core._utils import create_shared_memory_segment, destroy_shared_memory_segment

from .shared_memory import PublishedDataMatrixRecord, PublishedDataMatrixRecords

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory


# ==================================================================================================
#  DataMatrixAllocator
# ==================================================================================================
class DataMatrixAllocator(ABC):
    """A data matrix allocator is the interface through which a data matrix source places its data matrix."""

    @abstractmethod
    def allocate(self, matrix_id: int, shape: tuple[int, ...]) -> NDArray[np.float32]:
        """Return an uninitialized, writable float32 buffer of the given shape for the data matrix with the given id.

        Args:
            matrix_id: the id under which a shared-memory allocator publishes the data matrix, so that a
                worker process can find it; an in-process allocator ignores it.
            shape: the shape of the buffer.
        """

    @abstractmethod
    def adopt(self, matrix_id: int, array: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the array that a distance store will read for the data matrix with the given id.

        Args:
            matrix_id: the id under which a shared-memory allocator publishes the data matrix, so that a
                worker process can find it; an in-process allocator ignores it.
            array: the data matrix in its final form, which already exists: the user's vectors or
                distances, or vectors preprocessed for a lazy distance store's metric.
        """


# ==================================================================================================
#  InProcessDataMatrixAllocator
# ==================================================================================================
class InProcessDataMatrixAllocator(DataMatrixAllocator):
    """This allocator places data matrices in this process only, so it has no use for their matrix ids."""

    def allocate(self, matrix_id: int, shape: tuple[int, ...]) -> NDArray[np.float32]:
        """Return a plain numpy array of the given shape."""
        return np.empty(shape, dtype=np.float32)

    def adopt(self, matrix_id: int, array: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the array itself; nothing is copied."""
        return array


# ==================================================================================================
#  SharedMemoryDataMatrixAllocator
# ==================================================================================================
class SharedMemoryDataMatrixAllocator(DataMatrixAllocator):
    """This allocator places each data matrix in a shared-memory segment of its own, which worker processes can read.

    This process creates and owns every segment.  For each data matrix that it places, the
    allocator records, under the matrix id, a `PublishedDataMatrixRecord` that says which segment holds the
    matrix; `published_matrix_records` returns these records, which a worker process needs to find the
    segments.

    Place each matrix id at most once: placing an id again creates a second segment and replaces the
    record of the first, so a worker can no longer find the first segment.

    Closing the allocator destroys every segment that it created, which invalidates every distance
    store that reads one of them, in this process and in every worker process that attached.  Close
    the allocator only after every worker is done.
    """

    def __init__(self) -> None:
        """Start without any segment; segments are created as data matrices are allocated and adopted."""
        self._segments: list[SharedMemory] = []
        self._published_matrix_records: dict[int, PublishedDataMatrixRecord] = {}

    def allocate(self, matrix_id: int, shape: tuple[int, ...]) -> NDArray[np.float32]:
        """Create a segment sized for the given shape and return the writable array that views it."""
        return self._create_segment(matrix_id, shape)

    def adopt(self, matrix_id: int, array: NDArray[np.float32]) -> NDArray[np.float32]:
        """Copy the array into a new segment and return the array that views the segment."""
        buffer = self._create_segment(matrix_id, array.shape)
        buffer[:] = array
        return buffer

    @property
    def published_matrix_records(self) -> PublishedDataMatrixRecords:
        """Return the `PublishedDataMatrixRecord` of every data matrix that this allocator placed, by matrix id.

        The records stay available after `close`.
        """
        return PublishedDataMatrixRecords(dict(self._published_matrix_records))

    def close(self) -> None:
        """Destroy every segment that this allocator created.

        Every distance store that reads one of those segments becomes invalid, in this process and
        in every worker process that attached; call this only after every worker is done.
        """
        for segment in self._segments:
            destroy_shared_memory_segment(segment)
        self._segments.clear()

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    def _create_segment(self, matrix_id: int, shape: tuple[int, ...]) -> NDArray[np.float32]:
        """Create a shared-memory segment for the shape, publish it under the matrix id, and return its array."""
        segment = create_shared_memory_segment(PublishedDataMatrixRecord.nbytes_for(shape))
        self._segments.append(segment)
        published_matrix_record = PublishedDataMatrixRecord(segment_name=segment.name, shape=tuple(shape))
        self._published_matrix_records[matrix_id] = published_matrix_record
        return published_matrix_record.array_over(segment)
