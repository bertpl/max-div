"""This module lets a worker process read a distance store that another process built in shared memory.

A distance store keeps its data in one array, so one shared-memory segment holds that array.  The
process that builds the distance stores (through `SharedMemoryDistanceStoreAllocator`) creates the
segments and owns them.  For each distance store it records a `SharedStoreSpec`, a small record
that says which segment holds the array and how to rebuild the distance store over it.  A worker
process receives the specs as ordinary pickled arguments and calls `attached_distance_store` to
get a distance store that reads the segment's bytes.

A distance store that reads a shared-memory segment is an ordinary `DistanceStore`: the trackers
and the compiled functions downstream cannot tell it from one over a plain array.

The segment mechanics, and the lifetime rules that every user of a segment must follow, live in
`max_div._core._utils._shared_memory_segment`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from max_div._core._utils import attach_shared_memory_segment
from max_div._core.metrics._distance import KIND_FULL_MATRIX, DistanceMetric, DistanceStore


# ==================================================================================================
#  Specification
# ==================================================================================================
class SharedStoreSpec(NamedTuple):
    """A spec says which shared-memory segment holds a distance store's array and how to rebuild the store over it.

    It is small enough to travel to a worker process as an ordinary pickled argument.
    """

    segment_name: str  # the operating-system name of the segment, which is how another process finds it
    kind: int  # the `DistanceStore.kind` selector of the distance store that reads the segment
    shape: tuple[int, ...]  # the shape of the float32 array in the segment; its first axis is the item count
    distance_metric: DistanceMetric | None = (
        None  # the distance metric of a lazy distance store; None for a full matrix
    )

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def over_segment(
        cls,
        segment: SharedMemory,
        buffer: NDArray[np.float32],
        kind: np.int32,
        distance_metric: DistanceMetric | None = None,
    ) -> "SharedStoreSpec":
        """Return the spec that lets a worker process rebuild a distance store of the given kind over the segment."""
        return cls(segment_name=segment.name, kind=int(kind), shape=buffer.shape, distance_metric=distance_metric)

    # --------------------------------------------------------------------------
    #  Rebuilding the distance store
    # --------------------------------------------------------------------------
    def distance_store_over(self, buffer: NDArray[np.float32]) -> DistanceStore:
        """Return the distance store that reads the buffer as this spec's kind."""
        if self.kind == KIND_FULL_MATRIX:
            return DistanceStore.full_matrix(buffer)
        else:
            assert self.distance_metric is not None  # noqa: S101 -- the allocator gives every lazy spec its metric
            return DistanceStore.lazy(buffer, self.distance_metric)


# ==================================================================================================
#  Attaching
# ==================================================================================================
@contextmanager
def attached_distance_store(spec: SharedStoreSpec) -> Iterator[DistanceStore]:
    """Yield a distance store that reads the segment named in the spec, for the duration of the block.

    On exit this closes this process's mapping of the segment and never unlinks the segment, which
    belongs to the process that created it.
    """
    segment = attach_shared_memory_segment(spec.segment_name)
    try:
        yield spec.distance_store_over(np.ndarray(spec.shape, dtype=np.float32, buffer=segment.buf))
    finally:
        segment.close()
