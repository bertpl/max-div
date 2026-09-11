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
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from max_div._core._utils import attach_shared_memory_segment
from max_div._core.metrics._distance import KIND_FULL_MATRIX, DistanceMetric, DistanceStore


# =================================================================================================
#  Specification
# =================================================================================================
class SharedStoreSpec(NamedTuple):
    """A spec says which shared-memory segment holds a distance store's array and how to rebuild the store over it.

    It is small enough to travel to a worker process as an ordinary pickled argument.
    """

    segment_name: str  # the operating-system name of the segment, which is how another process finds it
    kind: int  # the `DistanceStore.kind` selector of the distance store that reads the segment
    metric_kind: int  # the distance metric that a lazy distance store computes with; unused for a full matrix
    metric_p: float  # `DistanceMetric.p` of that metric; unused for a full matrix
    shape: tuple[int, ...]  # the shape of the float32 array in the segment; its first axis is the item count


# =================================================================================================
#  Attaching
# =================================================================================================
@contextmanager
def attached_distance_store(spec: SharedStoreSpec) -> Iterator[DistanceStore]:
    """Yield a distance store that reads the segment named in the spec, for the duration of the block.

    On exit this closes this process's mapping of the segment and never unlinks the segment, which
    belongs to the process that created it.
    """
    segment = attach_shared_memory_segment(spec.segment_name)
    try:
        yield _store_over(np.ndarray(spec.shape, dtype=np.float32, buffer=segment.buf), spec)
    finally:
        segment.close()


def _store_over(buffer: NDArray[np.float32], spec: SharedStoreSpec) -> DistanceStore:
    """Return the distance store that reads the buffer as the kind that the spec names."""
    if spec.kind == KIND_FULL_MATRIX:
        return DistanceStore.full_matrix(buffer)
    return DistanceStore.lazy(buffer, DistanceMetric(kind=spec.metric_kind, p=spec.metric_p))
