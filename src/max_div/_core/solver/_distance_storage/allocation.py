"""An allocator decides where the arrays of a distance store are placed in memory.

The distance store factory decides what each distance store contains.  The allocator decides only
where the array that holds those contents lives: in this process, or in a shared-memory segment
that worker processes can read.  There is one allocator class per case, and the factory builds
every distance store the same way whichever allocator it is given.

The factory asks an allocator for two things:

- `allocate` returns an empty, writable buffer that the factory then fills, for example a full
  distance matrix that is computed straight into its final place.
- `adopt` takes an array that already exists in its final form, for example the user's own vectors,
  and returns the array that the distance store will read from.
"""

from abc import ABC, abstractmethod
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray

from max_div._core._utils import create_shared_memory_segment, destroy_shared_memory_segment
from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import NO_P

from .shared_memory import SharedStoreSpec


# =================================================================================================
#  DistanceStoreAllocator
# =================================================================================================
class DistanceStoreAllocator(ABC):
    """This is the interface through which the distance store factory obtains the arrays of its distance stores."""

    @abstractmethod
    def allocate(self, shape: tuple[int, ...], kind: np.int32) -> NDArray[np.float32]:
        """Return an uninitialized, writable float32 buffer of the given shape.

        The factory fills the buffer and then wraps it in a distance store of the given kind.

        Args:
            shape: the shape of the array to allocate.
            kind: the `DistanceStore.kind` selector of the distance store that will read the buffer.
        """

    @abstractmethod
    def adopt(self, array: NDArray[np.float32], kind: np.int32, metric: DistanceMetric | None) -> NDArray[np.float32]:
        """Return the array that a distance store of the given kind will read, for data that already exists.

        Args:
            array: the data in its final form: a full distance matrix, or the preprocessed vectors
                of a lazy distance store.
            kind: the `DistanceStore.kind` selector of the distance store that will read the array.
            metric: the distance metric that a lazy distance store computes with; None for a full
                distance matrix.
        """


# =================================================================================================
#  InProcessDistanceStoreAllocator
# =================================================================================================
class InProcessDistanceStoreAllocator(DistanceStoreAllocator):
    """This allocator allocates arrays in this process only; nothing is shared with other processes."""

    def allocate(self, shape: tuple[int, ...], kind: np.int32) -> NDArray[np.float32]:
        """Return a plain numpy array of the given shape."""
        return np.empty(shape, dtype=np.float32)

    def adopt(self, array: NDArray[np.float32], kind: np.int32, metric: DistanceMetric | None) -> NDArray[np.float32]:
        """Return the array itself; nothing is copied."""
        return array


# =================================================================================================
#  SharedMemoryDistanceStoreAllocator
# =================================================================================================
class SharedMemoryDistanceStoreAllocator(DistanceStoreAllocator):
    """This allocator allocates arrays in shared-memory segments, so that worker processes can read the same arrays.

    This process creates and owns every segment.  For each array that the factory allocates or
    adopts, the allocator records a `SharedStoreSpec` that says which segment holds the array and
    how to rebuild the distance store over it; a worker process attaches to the segment with that
    spec.  The specs are recorded in the order in which the factory asked for the arrays, which is
    the order of the distance stores.

    Adopting the same array twice puts it in one segment, not two.  This is how every lazy distance
    store whose metric reads the user's raw vectors shares a single copy of those vectors.

    Closing the allocator destroys every segment that it created, which invalidates every distance
    store that reads one of them, in this process and in every worker process that attached.  Close
    the allocator only after every worker is done.
    """

    def __init__(self) -> None:
        """Start without any segment; segments are created as the factory allocates and adopts arrays."""
        self._segments: list[SharedMemory] = []
        self._specs: list[SharedStoreSpec] = []
        self._segment_of_adopted: dict[int, tuple[SharedMemory, NDArray[np.float32]]] = {}  # keyed by id(array)

    def allocate(self, shape: tuple[int, ...], kind: np.int32) -> NDArray[np.float32]:
        """Create a segment sized for the given shape and return the writable array that views it."""
        segment, buffer = self._create_segment(shape)
        self._specs.append(_spec_for(segment, buffer, kind, None))
        return buffer

    def adopt(self, array: NDArray[np.float32], kind: np.int32, metric: DistanceMetric | None) -> NDArray[np.float32]:
        """Copy the array into a segment and return the array that views the segment.

        An array that was adopted before is not copied again: the array that views its existing
        segment is returned, and a second spec that names that segment is recorded.
        """
        known = self._segment_of_adopted.get(id(array))
        if known is None:
            segment, buffer = self._create_segment(array.shape)
            buffer[:] = array
            self._segment_of_adopted[id(array)] = (segment, buffer)
        else:
            segment, buffer = known
        self._specs.append(_spec_for(segment, buffer, kind, metric))
        return buffer

    @property
    def specs(self) -> tuple[SharedStoreSpec, ...]:
        """Return one spec per distance store that the factory built, in that order."""
        return tuple(self._specs)

    def close(self) -> None:
        """Destroy every segment that this allocator created.

        Every distance store that reads one of those segments becomes invalid, in this process and
        in every worker process that attached; call this only after every worker is done.
        """
        for segment in self._segments:
            destroy_shared_memory_segment(segment)
        self._segments.clear()
        self._segment_of_adopted.clear()

    def _create_segment(self, shape: tuple[int, ...]) -> tuple[SharedMemory, NDArray[np.float32]]:
        """Create a shared-memory segment for the given float32 shape and return it with the array that views it."""
        segment = create_shared_memory_segment(int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize)
        self._segments.append(segment)
        return segment, np.ndarray(shape, dtype=np.float32, buffer=segment.buf)


def _spec_for(
    segment: SharedMemory, buffer: NDArray[np.float32], kind: np.int32, metric: DistanceMetric | None
) -> SharedStoreSpec:
    """Return the spec that lets a worker process rebuild a distance store of the given kind over the segment."""
    return SharedStoreSpec(
        segment_name=segment.name,
        kind=int(kind),
        metric_kind=0 if metric is None else int(metric.kind),
        metric_p=NO_P if metric is None else float(metric.p),
        shape=buffer.shape,
    )
