"""A data matrix source says how a data matrix of a solve is produced: as an existing array, or computed into a buffer.

The source decides the contents of the data matrix, and the allocator that it is given decides
where the matrix lives; see `allocation`.  A source produces its matrix only when the data matrix
registry asks for it, so the factory can describe every data matrix, and check that the matrices
fit in memory, before any of them is computed.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .allocation import DataMatrixAllocator


# ==================================================================================================
#  DataMatrixSource
# ==================================================================================================
class DataMatrixSource(ABC):
    """A data matrix source produces one data matrix through an allocator, which decides where the matrix is placed."""

    @abstractmethod
    def produce(self, matrix_id: int, allocator: DataMatrixAllocator) -> NDArray[np.float32]:
        """Return the data matrix, placed by the allocator under the given matrix id."""


# ==================================================================================================
#  Sources
# ==================================================================================================
@dataclass(frozen=True, eq=False)
class ExistingDataMatrixSource(DataMatrixSource):
    """The data matrix exists already as an array, such as the user's vectors, and the allocator adopts it."""

    array: NDArray[np.float32]

    def produce(self, matrix_id: int, allocator: DataMatrixAllocator) -> NDArray[np.float32]:
        """Return the array as the allocator adopts it: the array itself in this process, a copy in shared memory."""
        return allocator.adopt(matrix_id, self.array)


@dataclass(frozen=True, eq=False)
class ComputedDataMatrixSource(DataMatrixSource):
    """The data matrix is computed straight into a buffer that the allocator allocates, so it is never copied."""

    shape: tuple[int, ...]
    compute_into: Callable[
        [NDArray[np.float32]], object
    ]  # compute_into writes the whole matrix into its buffer argument

    def produce(self, matrix_id: int, allocator: DataMatrixAllocator) -> NDArray[np.float32]:
        """Allocate a buffer of this source's shape, compute the matrix into it, and return it."""
        buffer = allocator.allocate(matrix_id, self.shape)
        self.compute_into(buffer)
        return buffer
