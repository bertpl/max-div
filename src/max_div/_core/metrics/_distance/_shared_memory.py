"""A distance store can be published into shared memory, so several processes read one copy of it.

A store populates exactly one of its arrays and leaves the others zero-length, so one segment holds
whichever array the backend uses.  Readers attach by name: the processes sharing a store are spawned
rather than forked, and inherit nothing.

A published store is not a second kind of store.  `SharedDistanceStore` owns the segment, and the
`DistanceStore` it hands out reads that segment like any other, so the trackers and the compiled
reads downstream cannot tell one from the other.

The whole sequence, across two processes:

  1. The publisher resolves a backend and calls `build_shared_distance_store`, which allocates a
     segment and builds the distances straight into it.
  2. It reads through `SharedDistanceStore.store`, and sends `SharedDistanceStore.spec` — a small
     picklable record — to each worker it spawns.
  3. A worker opens `attached_distance_store(spec)` and gets an ordinary `DistanceStore` over the
     same bytes, for the length of that block.
  4. Workers leave their blocks, each closing only its own mapping.
  5. The publisher closes last, destroying the segment.

`multiprocessing.shared_memory` is available on every platform the package supports.  What differs is
how long a segment lives, and POSIX leaves that to the processes, which imposes two obligations:

  - The publisher must outlive every reader, because it is the process that destroys the segment.
    A POSIX segment outlives its creator, and reading one through a closed mapping crashes rather
    than raising.
  - An attaching process must not register with CPython's resource tracker, which is shared by the
    whole process tree; `_attach_without_registering` covers why.  The publisher does register, and
    that registration releases the segment if the publisher dies holding it.

Windows has neither concern: it keeps no tracker, its `unlink` is documented as having no effect, and
the block goes away once the last handle closes.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ._store import KIND_FULL_MATRIX, KIND_LAZY, DistanceStore

# Whether SharedMemory accepts `track=False`.
_TRACK_FLAG_SUPPORTED = sys.version_info >= (3, 13)


# =================================================================================================
#  Specification
# =================================================================================================
class SharedStoreSpec(NamedTuple):
    """A spec records where a published store's bytes live and how to read them back.

    Small enough to travel to a worker process as an ordinary pickled argument.
    """

    segment_name: str  # OS-level name of the segment, which is how another process finds it
    kind: int  # which DistanceStore backend the segment's array holds data for
    n: int  # number of items the store covers
    metric_kind: int  # metric the lazy backend computes with; the stored backends ignore it
    metric_p: float  # the metric's power parameter, NaN for metrics without one
    shape: tuple[int, ...]  # shape of the float32 array in the segment, not of the problem


# =================================================================================================
#  Publishing
# =================================================================================================
class SharedDistanceStore:
    """A shared store keeps its distances in a segment this process created and owns.

    Closing unlinks the segment and invalidates every mapping of it, including those held by
    attached readers, so close only once the readers are done.
    """

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(
        self,
        segment: SharedMemory,
        shape: tuple[int, ...],
        kind: int,
        n: int,
        metric_kind: int,
        metric_p: float,
    ) -> None:
        """Wrap an owned segment; prefer `allocate`, which sizes the segment for the shape."""
        self._segment = segment
        self._buffer: NDArray[np.float32] = np.ndarray(shape, dtype=np.float32, buffer=segment.buf)
        self._spec = SharedStoreSpec(
            segment_name=segment.name,
            kind=int(kind),
            n=int(n),
            metric_kind=int(metric_kind),
            metric_p=float(metric_p),
            shape=shape,
        )

    @classmethod
    def allocate(
        cls, shape: tuple[int, ...], kind: int, n: int, metric_kind: int = 0, metric_p: float = float("nan")
    ) -> "SharedDistanceStore":
        """Create a segment sized for `shape` and return the owner reading from it.

        The buffer starts uninitialized: the caller fills it before publishing the spec.

        Args:
            shape: the buffer shape to size the segment for; its product is the float32 element count.
            kind: the DistanceStore backend selector the buffer holds data for.
            n: the number of items the distances are over.
            metric_kind: metric selector, meaningful for the lazy backend only.
            metric_p: the metric's power parameter, NaN when the metric has none.
        """
        # a zero-size segment is rejected by the OS, so degenerate shapes still claim one byte
        size_bytes = max(int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize, 1)
        return cls(SharedMemory(create=True, size=size_bytes), shape, kind, n, metric_kind, metric_p)

    # --------------------------------------------------------------------------
    #  Access
    # --------------------------------------------------------------------------
    @property
    def buffer(self) -> NDArray[np.float32]:
        """Return the writable view the data is built into, before any reader attaches."""
        return self._buffer

    @property
    def store(self) -> DistanceStore:
        """Return a DistanceStore reading this segment; the store factories make the view read-only."""
        return _store_over(self._buffer, self._spec)

    @property
    def spec(self) -> SharedStoreSpec:
        """Return what a reader needs to attach to this segment."""
        return self._spec

    # --------------------------------------------------------------------------
    #  Lifetime
    # --------------------------------------------------------------------------
    def close(self) -> None:
        """Unmap and destroy the segment; every store reading it becomes invalid."""
        # drop this object's segment-backed view first: reading one after close crashes
        self._buffer = np.empty(0, dtype=np.float32)
        self._segment.close()
        self._segment.unlink()

    def __enter__(self) -> "SharedDistanceStore":
        """Return the owner itself, so the segment is scoped to a `with` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Destroy the segment, invalidating every store reading it."""
        self.close()


def publish_distance_store(store: DistanceStore, n: int) -> SharedDistanceStore:
    """Copy an already-built store's data into a fresh segment and return the owner.

    For data that exists before a segment does — a distance matrix the caller supplied.  A store
    computed from vectors should instead be built straight into `SharedDistanceStore.allocate`'s
    buffer, which is the same bytes without the transient second copy.
    """
    array = _populated_array(store)
    shared = SharedDistanceStore.allocate(
        array.shape, int(store.kind), n, int(store.metric_kind), float(store.metric_p)
    )
    shared.buffer[:] = array
    return shared


def _populated_array(store: DistanceStore) -> NDArray[np.float32]:
    """Return the one array the store's backend holds its data in."""
    if store.kind == KIND_FULL_MATRIX:
        return store.matrix
    if store.kind == KIND_LAZY:
        return store.vectors
    return store.pdist


# =================================================================================================
#  Attaching
# =================================================================================================
@contextmanager
def attached_distance_store(spec: SharedStoreSpec) -> Iterator[DistanceStore]:
    """Yield a DistanceStore reading a published segment, for the duration of the block.

    Closes this process's mapping on exit and never unlinks: the segment belongs to whoever
    published it.
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
    """Attach with registration suppressed, doing what `track=False` does on Python 3.13 and later.

    Registering and then unregistering would be shorter and is wrong: one tracker daemon serves the
    whole process tree, so removing the entry removes the publisher's, and with it the cleanup that
    would have released the segment had the publisher died holding it.

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
    """Return the DistanceStore that reads `buffer` as the backend named in `spec`."""
    if spec.kind == KIND_FULL_MATRIX:
        return DistanceStore.full_matrix(buffer)
    if spec.kind == KIND_LAZY:
        return DistanceStore.lazy_prepared(buffer, np.int32(spec.metric_kind), np.float64(spec.metric_p))
    return DistanceStore.condensed(buffer, spec.n)
