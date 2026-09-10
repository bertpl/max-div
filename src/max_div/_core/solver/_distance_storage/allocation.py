"""An allocator decides where a store's array lives: in this process, or in a segment other processes can read.

The factory builds every store the same way whichever allocator it holds; only two calls differ:

- `allocate` hands out a writable buffer the factory fills — a full matrix computed or expanded
  straight into its final place, since at those sizes a build-then-copy would double peak resident
  memory for its duration.
- `adopt` takes an array that already exists in its final form, either as it is or copied into a
  segment, since the bytes a worker reads have to live there.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import NO_P

from .shared_memory import SharedDistanceStore, SharedStoreSpec


# =================================================================================================
#  StoreAllocator
# =================================================================================================
class StoreAllocator(ABC):
    """The interface the factory builds stores through; the module docstring states the contract."""

    @abstractmethod
    def allocate(self, shape: tuple[int, ...], kind: np.int32) -> NDArray[np.float32]:
        """Return an uninitialized writable float32 buffer of `shape`, to be filled and then wrapped as `kind`."""

    @abstractmethod
    def adopt(self, array: NDArray[np.float32], kind: np.int32, metric: DistanceMetric | None) -> NDArray[np.float32]:
        """Return the array the store of `kind` wraps for data that already exists in its final form.

        Args:
            array: the final-form data: a full matrix, or a lazy store's preprocessed vectors.
            kind: the `DistanceStore.kind` selector the store will carry.
            metric: the metric a lazy store reads with; None for a full matrix.
        """


# =================================================================================================
#  InProcessAllocator
# =================================================================================================
class InProcessAllocator(StoreAllocator):
    """Arrays live in this process: fresh buffers are plain numpy arrays, adopted arrays stay as they are."""

    def allocate(self, shape: tuple[int, ...], kind: np.int32) -> NDArray[np.float32]:
        """Return a plain numpy array; nothing is shared."""
        return np.empty(shape, dtype=np.float32)

    def adopt(self, array: NDArray[np.float32], kind: np.int32, metric: DistanceMetric | None) -> NDArray[np.float32]:
        """Return the array itself; nothing is copied."""
        return array


# =================================================================================================
#  SharedMemoryAllocator
# =================================================================================================
class SharedMemoryAllocator(StoreAllocator):
    """Arrays live in segments this process owns; the specs describe them to the processes that attach.

    `close` states the segments' lifetime.
    """

    def __init__(self) -> None:
        """Start with no segments; they are created as the factory allocates and adopts."""
        self._owners: list[SharedDistanceStore] = []
        self._specs: list[SharedStoreSpec] = []
        self._adopted: dict[int, SharedDistanceStore] = {}  # id(array) -> the segment holding its copy

    def allocate(self, shape: tuple[int, ...], kind: np.int32) -> NDArray[np.float32]:
        """Return the buffer of a fresh segment, recorded as one store."""
        owner = SharedDistanceStore.allocate(shape, kind)
        self._owners.append(owner)
        self._specs.append(owner.spec)
        return owner.buffer

    def adopt(self, array: NDArray[np.float32], kind: np.int32, metric: DistanceMetric | None) -> NDArray[np.float32]:
        """Return a segment's copy of the array, reusing the segment an earlier adoption of the same array made."""
        owner = self._adopted.get(id(array))
        if owner is None:
            metric_kind = np.int32(0) if metric is None else np.int32(metric.kind)
            metric_p = np.float64(NO_P if metric is None else metric.p)
            owner = SharedDistanceStore.allocate(array.shape, kind, metric_kind, metric_p)
            owner.buffer[:] = array
            self._owners.append(owner)
            self._adopted[id(array)] = owner
        self._specs.append(owner.spec)
        return owner.buffer

    @property
    def specs(self) -> tuple[SharedStoreSpec, ...]:
        """Return one spec per store the factory built, in that order; several may name one segment."""
        return tuple(self._specs)

    def close(self) -> None:
        """Destroy every segment, invalidating every store reading one, in this process and in every attached one.

        Close only once every reader is done.
        """
        for owner in self._owners:
            owner.close()
        self._owners.clear()
        self._adopted.clear()
