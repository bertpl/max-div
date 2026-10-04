"""A data matrix publisher produces the data matrices of a parallel solve in shared memory, for its worker processes.

The parent process of a parallel solve owns the shared-memory segments, and each worker reads them:

- `SharedMemoryDataMatrixPublisher` produces each data matrix in a shared-memory segment of its own,
  through `SharedMemoryDataMatrixAllocator`, and returns the `PublishedDataMatrixRecords` that
  locate the segments.
- A worker process receives those records as an ordinary pickled argument and reads the data
  matrices through a `SharedMemoryDataMatrixReader`.

A distance store over a shared-memory segment is an ordinary `DistanceStore`: the trackers and the
compiled functions downstream cannot tell it from one over a plain array.

The segment mechanics, and the lifetime rules that every user of a segment must follow, live in
`max_div._core._utils._shared_memory_segment`.
"""

from collections.abc import Mapping

from .allocation import PublishedDataMatrixRecords, SharedMemoryDataMatrixAllocator
from .data_matrix_producer import DataMatrixProducer


# ==================================================================================================
#  SharedMemoryDataMatrixPublisher
# ==================================================================================================
class SharedMemoryDataMatrixPublisher:
    """A shared-memory data matrix publisher produces a solve's data matrices in shared memory for a `with` block.

    Entering the block produces every data matrix in a shared-memory segment of its own and returns
    the records that locate the segments.  Leaving the block destroys the segments, which invalidates
    every distance store that reads one of them, in this process and in every worker; leave the block
    only after every worker is done.
    """

    def __init__(self, producers: Mapping[int, DataMatrixProducer]) -> None:
        """Keep the producers; the data matrices are produced when the block is entered."""
        self._producers = producers
        self._allocator = SharedMemoryDataMatrixAllocator()

    def __enter__(self) -> PublishedDataMatrixRecords:
        """Produce every data matrix in shared memory and return the records that locate them.

        If a producer raises, the segments created before it are destroyed and the exception propagates.
        """
        try:
            # this process reads none of the data matrices; they stay in their segments until the block ends
            DataMatrixProducer.produce_all(self._producers, self._allocator)
        except BaseException:
            self._allocator.close()
            raise
        return self._allocator.published_matrix_records

    def __exit__(self, *exc_info: object) -> None:
        """Destroy every segment that this publisher created."""
        self._allocator.close()
