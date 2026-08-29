"""An exchange slot holds one worker group's shared best: the top-scoring selection published so far.

A worker group — what the parallel-metaheuristics literature calls an *island* — is a set of
workers that exchange selections through one shared slot.  The slot lives in shared memory
created by the parent process and inherited by every group member at spawn, like the shared
distance store.  Everything in the slot is fixed-size, so it fits raw shared-memory arrays:

- the score, as a float64 vector;
- the selection, as an int32 vector of up to k indices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    from numpy.typing import NDArray


class GroupExchangeSlot:
    """The slot holds one worker group's shared best selection.

    Score tuples are compared lexicographically, exactly as `Score.as_tuple()` orders them, so
    "better" means the same thing here as in the final best-of-all selection.
    """

    def __init__(self, context: BaseContext, k: int, score_length: int) -> None:
        """Allocate the shared-memory fields; call in the parent, before workers spawn.

        Args:
            context: (BaseContext) the multiprocessing context the workers spawn from.
            k: (int) maximum selection size the slot can hold.
            score_length: (int) number of components in the workers' score tuples — a property
                of the shared metric configuration, so identical across a worker
                group's workers.
        """
        self._lock = context.Lock()
        # False until the first publish: an all-zero score array is a legal real score, so the
        # never-written state needs its own flag rather than a sentinel score
        self._written = context.Value("b", 0, lock=False)
        self._score = context.Array("d", score_length, lock=False)
        self._selection = context.Array("i", k, lock=False)
        self._n_selected = context.Value("i", 0, lock=False)

    def exchange(self, score: tuple[float, ...], selection: NDArray[np.int32]) -> NDArray[np.int32] | None:
        """Publish the given selection if it is strictly the group's best, else return a better one.

        The exchange is one atomic visit under the slot lock:

        - the worker's beats the stored one (or the slot was never written): the worker's is
          stored, and None is returned — the worker keeps what it has;
        - the stored one beats the worker's: a copy of the stored selection is returned for the
          worker to adopt;
        - equal scores: nothing is stored and None is returned.

        Args:
            score: (tuple[float, ...]) the worker's current score, as `Score.as_tuple()` orders it.
            selection: (int32 ndarray) the worker's current selection.

        Returns:
            the stored selection to adopt, or None to keep the worker's own.
        """
        with self._lock:
            if not self._written.value or score > tuple(self._score):
                self._score[:] = score
                self._selection[: len(selection)] = selection
                self._n_selected.value = len(selection)
                self._written.value = True
                return None
            if tuple(self._score) > score:
                return np.array(self._selection[: self._n_selected.value], dtype=np.int32)
            return None

    @property
    def written(self) -> bool:
        """Return whether any selection was ever stored."""
        with self._lock:
            return bool(self._written.value)

    def peek_score(self) -> tuple[float, ...] | None:
        """Return the stored score without touching the selection, or None for a never-written slot."""
        with self._lock:
            return tuple(self._score) if self._written.value else None
