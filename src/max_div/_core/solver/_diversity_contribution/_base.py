from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceStore


# =================================================================================================
#  PerItemContributionSource
# =================================================================================================
class PerItemContributionSource(ABC):
    """A source provides every item's per-item diversity contribution, the value the strategies sample items by.

    The *diversity contribution* of a point measures how much the point contributes to the diversity
    of the selection: for a selected point, how much it contributes to the current selection's
    diversity; for a non-selected point, how much diversity it would add if selected.  Higher is
    always more diverse.  A tracker is a source that maintains one spec's contribution itself; a
    hybrid source derives a hybrid objective's contribution from its term trackers on every read.
    """

    @abstractmethod
    def contribution_wrt_selection(self, selected: NDArray[np.bool], n_selected: np.int32) -> NDArray[np.floating]:
        """Return per-point contribution of all n points wrt the current selection.

        The returned array must be correct for selected and non-selected points alike, so callers
        can slice it by any selection mask.  It may be a reference to internal state — callers
        must not modify it.

        Args:
            selected: (n-sized bool ndarray) current selection mask.
            n_selected: (np.int32) number of True values in `selected`.
        """
        raise NotImplementedError


# =================================================================================================
#  DiversityContributionTracker
# =================================================================================================
class DiversityContributionTracker(PerItemContributionSource):
    """Tracks each point's per-point diversity contribution wrt an incrementally changing selection.

    Depending on the tracker family, the tracked value is the point's exact marginal contribution to
    the diversity objective or a monotone proxy for it.  Each concrete tracker defines the
    contribution of one diversity-metric family and owns the arrays and the numba-compiled functions
    that maintain it incrementally.

    Mutations mirror the solver-state mutators (`add`, `remove`, `..._many`) and must be called
    with the same indices, in the same order.  Snapshot methods mirror the solver-state snapshot
    life cycle, which is a *stack*: `push_snapshot` saves the current contributions on top of any
    already saved, and `pop_snapshot` discards the top entry, restoring from it or not.  The
    numba-compiled functions are only ever handed bare numpy arrays, never tracker objects.
    """

    # -------------------------------------------------------------------------
    #  Store
    # -------------------------------------------------------------------------
    @property
    @abstractmethod
    def store(self) -> DistanceStore:
        """Return the distance store this tracker reads (shared, immutable)."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    #  Mutations
    # -------------------------------------------------------------------------
    @abstractmethod
    def add(self, index: np.int32) -> None:
        """Update contributions after adding point `index` to the selection."""
        raise NotImplementedError

    @abstractmethod
    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update contributions after removing point `index` from the selection.

        Args:
            index: (np.int32) the point just removed.
            new_selection: (int32 ndarray) indices selected *after* the removal — needed by
                trackers whose contribution requires rescanning the remaining selection.
        """
        raise NotImplementedError

    def remove_trial(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update the contributions of the points in `new_selection` after removing point `index`.

        For a removal that is only scored and then reverted: the score reads the selected points
        only, so the other points' contributions may be left stale; that staleness is the saving
        over `remove`.  The caller must restore a snapshot taken before the call.  This default
        performs the full `remove`, which is correct but not cheaper.
        """
        self.remove(index, new_selection)

    def add_many(self, indices: NDArray[np.int32], parallel: bool = False) -> None:
        """Update contributions after adding all points in `indices` to the selection.

        `parallel` lets the update run over parallel threads where a tracker implements that;
        results are identical either way, and this default implementation ignores the flag.
        Callers may only opt in when no other worker process is competing for the CPU cores —
        inside one of several concurrently solving workers, the threads would oversubscribe them.
        """
        for index in indices:
            self.add(index)

    def remove_many(self, indices: NDArray[np.int32], new_selection: NDArray[np.int32]) -> None:
        """Update contributions after removing all points in `indices` from the selection.

        Args:
            indices: (int32 ndarray) the points just removed.
            new_selection: (int32 ndarray) indices selected after *all* removals.
        """
        for index in indices:
            self.remove(index, new_selection)

    @abstractmethod
    def reset(self) -> None:
        """Reset contributions to the empty selection, without going through per-point removes.

        Callers must only reset when the snapshot stack is empty: a reset does not touch saved
        snapshots, so after a reset inside an open snapshot scope, a later restore would bring
        back contributions that no longer match the rest of the solver state.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    #  Snapshot
    # -------------------------------------------------------------------------
    @abstractmethod
    def push_snapshot(self) -> None:
        """Save the current contribution state on top of the snapshot stack."""
        raise NotImplementedError

    @abstractmethod
    def pop_snapshot(self, restore: bool) -> None:
        """Discard the top snapshot, first restoring the contribution state from it if `restore`.

        Args:
            restore: (bool) True to restore the snapshotted state, False to keep the
                current state and drop the snapshot.
        """
        raise NotImplementedError
