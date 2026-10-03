"""A data matrix registry holds the data matrices of one solve, each produced from its source, by matrix id."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics._distance import DataMatrixReader

from .allocation import DataMatrixAllocator, InProcessDataMatrixAllocator, SharedMemoryDataMatrixAllocator
from .data_matrix_source import DataMatrixSource
from .shared_memory import PublishedDataMatrices


# ==================================================================================================
#  DataMatrixRegistry
# ==================================================================================================
class DataMatrixRegistry(DataMatrixReader):
    """A data matrix registry produces the data matrices of one solve from their sources and returns each by its id.

    Each data matrix is produced once, so 2 distance stores that read the same matrix id read the
    same array.  Create a registry with `in_process`, or publish the data matrices to shared memory
    with `published_to_shared_memory`.
    """

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(self, sources: Mapping[int, DataMatrixSource], allocator: DataMatrixAllocator) -> None:
        """Produce every data matrix from its source, placed by the allocator.

        Args:
            sources: the source of each data matrix to produce, by matrix id; a matrix that no
                distance store reads has no source, so it is never produced.
            allocator: decides where each data matrix is placed.
        """
        self._arrays = {matrix_id: source.produce(matrix_id, allocator) for matrix_id, source in sources.items()}

    @classmethod
    def in_process(cls, sources: Mapping[int, DataMatrixSource]) -> "DataMatrixRegistry":
        """Return the registry of the data matrices produced in this process, which nothing else can read."""
        return cls(sources, InProcessDataMatrixAllocator())

    @classmethod
    @contextmanager
    def published_to_shared_memory(cls, sources: Mapping[int, DataMatrixSource]) -> Iterator[PublishedDataMatrices]:
        """Produce the data matrices in shared memory and yield their published records, for the duration of the block.

        This is a context manager.  Inside the block the segments exist and worker processes can
        attach to them with the yielded records.  On exit the segments are destroyed, so leave the
        block only after every worker is done.
        """
        allocator = SharedMemoryDataMatrixAllocator()
        try:
            # this process reads none of the matrices; they stay in their segments until the allocator closes
            cls(sources, allocator)
            yield allocator.published
        finally:
            allocator.close()

    # --------------------------------------------------------------------------
    #  Reading
    # --------------------------------------------------------------------------
    def array(self, matrix_id: int) -> NDArray[np.float32]:
        """Return the data matrix with the given id."""
        return self._arrays[matrix_id]
