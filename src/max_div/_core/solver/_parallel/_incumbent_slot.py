"""One shared incumbent per island: the best selection any island member has published so far.

The slot lives in shared memory created by the parent process and inherited by every island
member at spawn, like the shared distance store.  Everything in it is fixed-size, so it fits raw
shared-memory arrays: the score as a float64 vector (its length is a property of the shared
metric configuration, so it is identical across an island's workers), the selection as an int32
vector of up to k indices.  A version counter distinguishes a never-written slot from a written
one and lets tests observe that exchanges happened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    from numpy.typing import NDArray


class IslandIncumbentSlot:
    """The shared best-selection slot of one island, exchanged through under a single lock.

    Score tuples are compared lexicographically, exactly as `Score.as_tuple()` orders them, so
    "better" means the same thing here as in the final best-of-all selection.
    """

    def __init__(self, context: BaseContext, k: int, score_length: int) -> None:
        """Allocate the shared-memory fields; call in the parent, before workers spawn.

        :param context: (BaseContext) the multiprocessing context the workers spawn from.
        :param k: (int) maximum selection size the slot can hold.
        :param score_length: (int) number of components in the workers' score tuples.
        """
        self._lock = context.Lock()
        self._version = context.Value("q", 0, lock=False)
        self._score = context.Array("d", score_length, lock=False)
        self._selection = context.Array("i", k, lock=False)
        self._n_selected = context.Value("i", 0, lock=False)

    def exchange(self, score: tuple[float, ...], selection: NDArray[np.int32]) -> NDArray[np.int32] | None:
        """Publish the given selection if it is strictly the island's best, else return a better one.

        One atomic visit under the slot lock, from which a worker comes away with whichever of the
        two selections scores strictly higher:

        - the worker's beats the stored one (or the slot was never written): the worker's is
          stored, and None is returned — the worker keeps what it has;
        - the stored one beats the worker's: a copy of the stored selection is returned for the
          worker to adopt;
        - equal scores: nothing is stored and None is returned.

        :param score: (tuple[float, ...]) the worker's current score, as `Score.as_tuple()` orders it.
        :param selection: (int32 ndarray) the worker's current selection.
        :returns: the stored selection to adopt, or None to keep the worker's own.
        """
        with self._lock:
            if self._version.value == 0 or score > tuple(self._score):
                self._score[:] = score
                self._selection[: len(selection)] = selection
                self._n_selected.value = len(selection)
                self._version.value += 1
                return None
            if tuple(self._score) > score:
                return np.array(self._selection[: self._n_selected.value], dtype=np.int32)
            return None

    @property
    def version(self) -> int:
        """Return how often a selection was stored; 0 means the slot was never written."""
        with self._lock:
            return self._version.value
