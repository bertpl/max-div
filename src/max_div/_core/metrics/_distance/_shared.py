"""Publishing a distance store into shared memory, so several processes read one copy of it.

A store populates exactly one of its arrays and leaves the others zero-length, so publication does
not branch per backend: allocate a segment, build into it, and hand out the segment name together
with the scalars needed to rebuild the same store elsewhere.  Readers attach by name, because the
processes that share a store are spawned rather than forked and inherit nothing.

Two properties of POSIX shared memory shape the rest:

  - A segment outlives the process that created it.  So the publisher unlinks it and attaching
    processes close only their own mapping — and the publisher must outlive every reader, since
    reading through a closed mapping is a use-after-unmap rather than an error.
  - CPython's resource tracker registers every segment a process touches and unlinks what it holds
    at exit, which for an attaching process means destroying a segment its owner is still lending
    out.  Python 3.13 added `track=False` to opt out; earlier versions undo the registration
    directly.  The publisher stays registered on purpose: that registration is what releases the
    segment if the owning process dies without cleaning up.
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

# Whether SharedMemory can attach without registering with the resource tracker (Python 3.13+).
_TRACK_FLAG_SUPPORTED = sys.version_info >= (3, 13)


# =================================================================================================
#  Specification
# =================================================================================================
class SharedStoreSpec(NamedTuple):
    """Where a published store's bytes live, and how to read them as a DistanceStore.

    Holds nothing but a name, a shape and the store's own scalars, so it travels to a worker
    process as an ordinary pickled argument.
    """

    segment_name: str
    kind: int
    n: int
    metric_kind: int
    shape: tuple[int, ...]


# =================================================================================================
#  Publishing
# =================================================================================================
class SharedDistanceStore:
    """A distance store whose data lives in a shared-memory segment this process owns.

    Allocate it, fill `buffer` once, then read through `store` and send `spec` to the processes
    that should share it.  Closing unlinks the segment and invalidates every mapping of it,
    including those held by attached readers, so close only once the readers are done.
    """

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(self, segment: SharedMemory, shape: tuple[int, ...], kind: int, n: int, metric_kind: int) -> None:
        """Wrap an owned segment; prefer `allocate`, which sizes the segment for the shape."""
        self._segment = segment
        self._buffer: NDArray[np.float32] = np.ndarray(shape, dtype=np.float32, buffer=segment.buf)
        self._spec = SharedStoreSpec(
            segment_name=segment.name, kind=int(kind), n=int(n), metric_kind=int(metric_kind), shape=shape
        )

    @classmethod
    def allocate(cls, shape: tuple[int, ...], kind: int, n: int, metric_kind: int = 0) -> "SharedDistanceStore":
        """Create a segment sized for `shape` and return the owner reading from it.

        The buffer starts uninitialized: the caller fills it before publishing the spec.

        :param shape: shape of the float32 array this backend stores.
        :param kind: the DistanceStore backend selector the buffer holds data for.
        :param n: number of items.
        :param metric_kind: pair-kernel selector, meaningful for the lazy backend only.
        """
        # a zero-size segment is rejected by the OS, so degenerate shapes still claim one byte
        size_bytes = max(int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize, 1)
        return cls(SharedMemory(create=True, size=size_bytes), shape, kind, n, metric_kind)

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
        self._buffer = np.empty(self._spec.shape, dtype=np.float32)[:0]
        self._segment.close()
        self._segment.unlink()

    def __enter__(self) -> "SharedDistanceStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def publish(store: DistanceStore, n: int) -> SharedDistanceStore:
    """Copy an already-built store's data into a fresh segment and return the owner.

    For data that exists before a segment does — a distance matrix the caller supplied.  A store
    computed from vectors should instead be built straight into `SharedDistanceStore.allocate`'s
    buffer, which is the same bytes without the transient second copy.
    """
    array = _populated_array(store)
    shared = SharedDistanceStore.allocate(array.shape, int(store.kind), n, int(store.metric_kind))
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

    Registering and then unregistering would be the shorter route and is wrong: one tracker daemon
    serves the whole process tree, so removing the entry removes the *publisher's* — and with it the
    cleanup that would have released the segment had the publisher died holding it.  Suppressing the
    registration instead leaves the publisher's entry alone.

    The suppression is process-wide for the length of one constructor call, so a thread creating a
    segment at that exact moment would lose its own registration.  Attaching happens once, at worker
    start-up, before there is anything else to race with.  Windows keeps no tracker.
    """
    if sys.platform == "win32":
        return SharedMemory(name=segment_name)
    registered = resource_tracker.register
    # replacing the module's bound method is the suppression itself, hence the assignment ty flags
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
        return DistanceStore.lazy_prepared(buffer, np.int32(spec.metric_kind))
    return DistanceStore.condensed(buffer, spec.n)
