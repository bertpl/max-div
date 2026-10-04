"""A data matrix producer says how a solve's data matrix is produced: as an existing array, or computed into a buffer.

The producer decides the contents of the data matrix, and the data matrix allocator that it is
given decides where the matrix lives; see `allocation`.

A producer produces its matrix only when a data matrix reader or publisher runs it, so the distance
storage plan can describe every data matrix, and check that the matrices fit in memory, before any
of them is computed.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .allocation import DataMatrixAllocator
from .memory_budget import data_matrix_bytes


# ==================================================================================================
#  DataMatrixProducer
# ==================================================================================================
class DataMatrixProducer(ABC):
    """A data matrix producer produces one data matrix through an allocator, which decides where the matrix lives."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data matrix; its first axis is the item count."""

    @abstractmethod
    def produce(self, matrix_id: int, allocator: DataMatrixAllocator) -> NDArray[np.float32]:
        """Return the data matrix, placed by the allocator under the given matrix id."""

    @abstractmethod
    def bytes_allocated(self, is_adopted_array_copied: bool) -> int:
        """Return the bytes that producing the data matrix allocates.

        Args:
            is_adopted_array_copied: whether the allocator copies an array that it adopts, as the
                shared-memory allocator does.
        """

    @property
    def nbytes(self) -> int:
        """Return the size in bytes of the float32 data matrix."""
        return data_matrix_bytes(self.shape)

    @staticmethod
    def produce_all(
        producers: Mapping[int, "DataMatrixProducer"], allocator: DataMatrixAllocator
    ) -> dict[int, NDArray[np.float32]]:
        """Return every data matrix by matrix id, each produced once by its producer through the allocator."""
        return {matrix_id: producer.produce(matrix_id, allocator) for matrix_id, producer in producers.items()}


# ==================================================================================================
#  Kinds of producer
# ==================================================================================================
@dataclass(frozen=True, eq=False)
class AdoptingDataMatrixProducer(DataMatrixProducer):
    """An adopting data matrix producer hands an existing array, such as the user's vectors, to the allocator."""

    array: NDArray[np.float32]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array."""
        return self.array.shape

    def produce(self, matrix_id: int, allocator: DataMatrixAllocator) -> NDArray[np.float32]:
        """Return the array as the allocator adopts it."""
        return allocator.adopt(matrix_id, self.array)

    def bytes_allocated(self, is_adopted_array_copied: bool) -> int:
        """Return the size of the array when the allocator copies it, and 0 when it does not."""
        if is_adopted_array_copied:
            return self.nbytes
        else:
            return 0


@dataclass(frozen=True, eq=False, slots=True)
class ComputingDataMatrixProducer(DataMatrixProducer):
    """A computing data matrix producer computes its matrix into a buffer from the allocator, so it is never copied."""

    # `slots=True` turns this field into a slot attribute on the generated class, and that attribute
    # overrides the abstract `shape` property inherited from `DataMatrixProducer`.  The explicit
    # `field()` keeps the field required: without it, the dataclass would take the inherited
    # property object as the field's default.
    shape: tuple[int, ...] = field()
    # compute_into writes the whole matrix into its buffer argument
    compute_into: Callable[[NDArray[np.float32]], object]

    def produce(self, matrix_id: int, allocator: DataMatrixAllocator) -> NDArray[np.float32]:
        """Allocate a buffer of this producer's shape, compute the matrix into it, and return it."""
        buffer = allocator.allocate(matrix_id, self.shape)
        self.compute_into(buffer)
        return buffer

    def bytes_allocated(self, is_adopted_array_copied: bool) -> int:
        """Return the size of the buffer, which every allocator allocates."""
        return self.nbytes
