"""This module lets a worker process read the data matrices that another process published in shared memory.

The process that publishes the data matrices owns their segments, and a worker process reads them:

- `SharedMemoryDataMatrixAllocator` places each data matrix in a shared-memory segment of its own
  and records a `PublishedDataMatrix` per matrix id, which says which segment holds that matrix.
- A worker process receives the `PublishedDataMatrices` as an ordinary pickled argument.
- The worker reads the data matrices through an `AttachedDataMatrixRegistry`.

`PublishedDistanceStores` bundles the `PublishedDataMatrices` with the distance spec of each
distance store, in store order, so that a worker process builds the same distance stores that
`DistanceStoreFactory.create_stores` builds in a single process.

A distance store over a shared-memory segment is an ordinary `DistanceStore`: the trackers and the
compiled functions downstream cannot tell it from one over a plain array.

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

    def array_over(self, segment: "SharedMemory") -> NDArray[np.float32]:
        """Return the float32 array of this data matrix's shape over the segment's bytes."""
        return np.ndarray(self.shape, dtype=np.float32, buffer=segment.buf)


@dataclass(frozen=True)
class PublishedDataMatrices:
    """Published data matrices hold, by matrix id, what another process needs to find the data matrices of one solve."""

    matrices: dict[int, PublishedDataMatrix]


# ==================================================================================================
#  AttachedDataMatrixRegistry
# ==================================================================================================
class AttachedDataMatrixRegistry(DataMatrixReader):
    """An attached data matrix registry reads the data matrices that another process published, by matrix id.

    Entering the block attaches to every published segment; leaving closes this process's mapping of
    each segment and never unlinks a segment, which belongs to the process that
    published it.  Read the data matrices, and the distance stores over them, only inside the block.
    """

    def __init__(self, published_matrices: PublishedDataMatrices) -> None:
        """Keep the published data matrices; the segments are attached when the block is entered."""
        self._published_matrices = published_matrices
        self._segments: list[SharedMemory] = []
        self._arrays: dict[int, NDArray[np.float32]] = {}

    def __enter__(self) -> "AttachedDataMatrixRegistry":
        """Attach to every published segment and return this registry.

        Raises:
            FileNotFoundError: If a segment no longer exists; the segments attached before it are closed.
        """
        try:
            for matrix_id, published_matrix in self._published_matrices.matrices.items():
                segment = attach_shared_memory_segment(published_matrix.segment_name)
                self._segments.append(segment)
                self._arrays[matrix_id] = published_matrix.array_over(segment)
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
    """Published distance stores hold a solve's data matrices in shared memory and the distance spec of each store.

    It is small enough to pass to a worker process as an ordinary pickled argument.
    """

    published_matrices: PublishedDataMatrices
    distance_specs: tuple[DistanceSpec, ...]  # there is one distance spec per distance store, in store order

    @contextmanager
    def attached_distance_stores(self) -> Iterator[list[DistanceStore]]:
        """Yield the distance stores, in store order, over the attached data matrices, for the duration of the block."""
        with AttachedDataMatrixRegistry(self.published_matrices) as data_matrix_reader:
            yield [spec.build_distance_store(data_matrix_reader) for spec in self.distance_specs]
