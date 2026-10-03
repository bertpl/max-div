"""An allocator decides where each data matrix of a solve is placed in memory.

A data matrix is an array that a distance store reads: a full distance matrix, or the vectors
that a lazy distance store computes its distances from.  Each data matrix of a solve has a matrix
id.  A `DataMatrixSource` decides what a data matrix contains; the allocator decides only where it
lives: in this process, or in a shared-memory segment that worker processes can read.  There is
one allocator class per case, and a source produces its data matrix the same way whichever
allocator it is given.

A source asks an allocator for one of two things:

- `allocate` returns an empty, writable buffer that the source then fills, for example a full
  distance matrix that is computed straight into its final place.
- `adopt` takes an array that already exists in its final form, for example the user's own vectors,
  and returns the array that a distance store will read from.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from max_div._core._utils import create_shared_memory_segment, destroy_shared_memory_segment

from .shared_memory import PublishedDataMatrices, PublishedDataMatrix

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory


# ==================================================================================================
#  DataMatrixAllocator
# ==================================================================================================
class DataMatrixAllocator(ABC):
    """This is the interface through which a data matrix source places its data matrix."""

    @abstractmethod
    def allocate(self, matrix_id: int, shape: tuple[int, ...]) -> NDArray[np.float32]:
        """Return an uninitialized, writable float32 buffer of the given shape for the data matrix with the given id."""

    @abstractmethod
    def adopt(self, matrix_id: int, array: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the array that a distance store will read for the data matrix with the given id.

        Args:
            matrix_id: the id of the data matrix that `array` holds.
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
    allocator records a `PublishedDataMatrix` under the matrix id, which says which segment holds
    the matrix; `published` returns these records, which a worker process needs to find the
    segments.

    Closing the allocator destroys every segment that it created, which invalidates every distance
    store that reads one of them, in this process and in every worker process that attached.  Close
    the allocator only after every worker is done.
    """

    def __init__(self) -> None:
        """Start without any segment; segments are created as data matrices are allocated and adopted."""
        self._segments: list[SharedMemory] = []
        self._published_matrices: dict[int, PublishedDataMatrix] = {}

    def allocate(self, matrix_id: int, shape: tuple[int, ...]) -> NDArray[np.float32]:
        """Create a segment sized for the given shape and return the writable array that views it."""
        return self._create_segment(matrix_id, shape)

    def adopt(self, matrix_id: int, array: NDArray[np.float32]) -> NDArray[np.float32]:
        """Copy the array into a new segment and return the array that views the segment."""
        buffer = self._create_segment(matrix_id, array.shape)
        buffer[:] = array
        return buffer

    @property
    def published(self) -> PublishedDataMatrices:
        """Return the published data matrix of every matrix id that this allocator placed.

        The records stay available after `close`, as a record of what was placed.
        """
        return PublishedDataMatrices(dict(self._published_matrices))

    def close(self) -> None:
        """Destroy every segment that this allocator created.

        Every distance store that reads one of those segments becomes invalid, in this process and
        in every worker process that attached; call this only after every worker is done.
        """
        for segment in self._segments:
            destroy_shared_memory_segment(segment)
        self._segments.clear()

    def _create_segment(self, matrix_id: int, shape: tuple[int, ...]) -> NDArray[np.float32]:
        """Create a segment for the float32 shape, publish it under the matrix id, and return an array over it."""
        segment = create_shared_memory_segment(int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize)
        self._segments.append(segment)
        self._published_matrices[matrix_id] = PublishedDataMatrix(segment_name=segment.name, shape=tuple(shape))
        return np.ndarray(shape, dtype=np.float32, buffer=segment.buf)
