"""This module lets a worker process read the data matrices that another process published in shared memory.

The process that builds the distance stores places each data matrix in a shared-memory segment of
its own, through `SharedMemoryDataMatrixAllocator`, and owns the segments.  The allocator records a
`PublishedDataMatrix` per matrix id, which says which segment holds that data matrix.  A worker
process receives the `PublishedDataMatrices` as an ordinary pickled argument and reads them through
an `AttachedDataMatrixRegistry`.

`PublishedDistanceStores` adds the distance spec of each distance store, in store order, so that a
worker process builds the same distance stores as the process that published them.  A distance
store over a shared-memory segment is an ordinary `DistanceStore`: the trackers and the compiled
functions downstream cannot tell it from one over a plain array.

The segment mechanics, and the lifetime rules that every user of a segment must follow, live in
`max_div._core._utils._shared_memory_segment`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from max_div._core._utils import attach_shared_memory_segment
from max_div._core.metrics._distance import DataMatrixReader, DistanceSpec, DistanceStore

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory


# ==================================================================================================
#  Published data matrices
# ==================================================================================================
class PublishedDataMatrix(NamedTuple):
    """A published data matrix says which shared-memory segment holds one data matrix, and its shape."""

    segment_name: str  # the operating-system name of the segment, which is how another process finds it
    shape: tuple[int, ...]  # the shape of the float32 array in the segment; its first axis is the item count


@dataclass(frozen=True)
class PublishedDataMatrices:
    """The published data matrices of one solve, by matrix id: everything another process needs to find them."""

    matrices: dict[int, PublishedDataMatrix]


# ==================================================================================================
#  AttachedDataMatrixRegistry
# ==================================================================================================
class AttachedDataMatrixRegistry(DataMatrixReader):
    """An attached data matrix registry reads the data matrices that another process published, by matrix id.

    It is a context manager.  Entering attaches to every published segment; leaving closes this
    process's mapping of each segment and never unlinks a segment, which belongs to the process that
    published it.  Read the data matrices, and the distance stores over them, only inside the block.
    """

    def __init__(self, published: PublishedDataMatrices) -> None:
        """Keep the published data matrices; the segments are attached when the block is entered."""
        self._published = published
        self._segments: list[SharedMemory] = []
        self._arrays: dict[int, NDArray[np.float32]] = {}

    def __enter__(self) -> "AttachedDataMatrixRegistry":
        """Attach to every published segment and return this registry.

        Raises:
            FileNotFoundError: If a segment no longer exists; the segments attached before it are closed.
        """
        try:
            for matrix_id, matrix in self._published.matrices.items():
                segment = attach_shared_memory_segment(matrix.segment_name)
                self._segments.append(segment)
                self._arrays[matrix_id] = np.ndarray(matrix.shape, dtype=np.float32, buffer=segment.buf)
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

    def _close(self) -> None:
        """Close this process's mapping of every attached segment, and forget the arrays over them."""
        self._arrays.clear()
        for segment in self._segments:
            segment.close()
        self._segments.clear()


# ==================================================================================================
#  PublishedDistanceStores
# ==================================================================================================
@dataclass(frozen=True)
class PublishedDistanceStores:
    """The published distance stores of one solve: the data matrices in shared memory, and each store's distance spec.

    It is small enough to travel to a worker process as an ordinary pickled argument.
    """

    data_matrices: PublishedDataMatrices
    distance_specs: tuple[DistanceSpec, ...]  # one per distance store, in store order

    @contextmanager
    def attached_distance_stores(self) -> Iterator[list[DistanceStore]]:
        """Yield the distance stores, in store order, over the attached data matrices, for the duration of the block."""
        with AttachedDataMatrixRegistry(self.data_matrices) as data_matrix_reader:
            yield [spec.build_distance_store(data_matrix_reader) for spec in self.distance_specs]
