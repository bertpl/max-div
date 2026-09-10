"""This module lets a worker process read a distance store that another process built in shared memory.

A distance store keeps its data in one array, so one shared-memory segment holds that array.  The
process that builds the distance stores (through `SharedMemoryDistanceStoreAllocator`) creates the
segments and owns them.  For each distance store it records a `SharedStoreSpec`, a small record
that says which segment holds the array and how to rebuild the distance store over it.  A worker
process receives the specs as ordinary pickled arguments and calls `attached_distance_store` to
get a distance store that reads the segment's bytes.

A distance store that reads a shared-memory segment is an ordinary `DistanceStore`: the trackers
and the compiled functions downstream cannot tell it from one over a plain array.

`multiprocessing.shared_memory` is available on every platform that the package supports.  What
differs is how long a segment lives.  POSIX leaves that to the processes, which imposes two
obligations:

- The process that created a segment must outlive every reader, because it is the process that
  destroys the segment.  A POSIX segment outlives its creator, and reading one through a closed
  mapping crashes rather than raising.
- An attaching process must not register with CPython's resource tracker, which is shared by the
  whole process tree; `_attach_without_registering` explains why.  The creating process does
  register, and that registration releases the segment if the creating process dies holding it.

Windows has neither concern: it keeps no tracker, its `unlink` is documented as having no effect,
and a segment goes away once the last handle to it closes.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics._distance import KIND_FULL_MATRIX, DistanceMetric, DistanceStore

# Whether SharedMemory accepts `track=False`.
_TRACK_FLAG_SUPPORTED = sys.version_info >= (3, 13)


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
    segment = _attach_untracked(spec.segment_name)
    try:
        yield _store_over(np.ndarray(spec.shape, dtype=np.float32, buffer=segment.buf), spec)
    finally:
        segment.close()


def _attach_untracked(segment_name: str) -> SharedMemory:
    """Attach to an existing segment without becoming responsible for destroying it."""
    if _TRACK_FLAG_SUPPORTED:
        return SharedMemory(name=segment_name, track=False)
    return _attach_without_registering(segment_name)


def _attach_without_registering(segment_name: str) -> SharedMemory:
    """Attach with registration suppressed, which is what `track=False` does on Python 3.13 and later.

    Registering and then unregistering would be shorter and is wrong: one tracker daemon serves the
    whole process tree, so removing the entry removes the creating process's entry too, and with it
    the cleanup that would have released the segment had the creating process died holding it.

    The suppression is process-wide for the length of one constructor call, so callers must not
    attach while another thread is creating a segment.  Windows keeps no tracker.
    """
    if sys.platform == "win32":
        return SharedMemory(name=segment_name)
    registered = resource_tracker.register
    # ty flags the assignment; replacing the module's bound method is the suppression itself
    resource_tracker.register = lambda *args, **kwargs: None  # ty: ignore[invalid-assignment]
    try:
        return SharedMemory(name=segment_name)
    finally:
        resource_tracker.register = registered


def _store_over(buffer: NDArray[np.float32], spec: SharedStoreSpec) -> DistanceStore:
    """Return the distance store that reads the buffer as the kind that the spec names."""
    if spec.kind == KIND_FULL_MATRIX:
        return DistanceStore.full_matrix(buffer)
    return DistanceStore.lazy(buffer, DistanceMetric(kind=spec.metric_kind, p=spec.metric_p))
