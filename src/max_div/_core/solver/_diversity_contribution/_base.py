from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


# =================================================================================================
#  DiversityContributionTracker
# =================================================================================================
class DiversityContributionTracker(ABC):
    """Tracks each point's per-point diversity contribution wrt an incrementally changing selection.

    The *diversity contribution* of a point measures how much the point contributes to the diversity
    of the selection: for a selected point, how much it contributes to the current selection's
    diversity; for a non-selected point, how much diversity it would add if selected.  Higher is
    always more diverse.  Depending on the tracker family, the tracked value is the point's exact
    marginal contribution to the diversity objective or a monotone proxy for it.  Each concrete
    tracker defines the contribution of one diversity-metric family and owns the arrays + kernel
    calls that maintain it incrementally.

    Mutations mirror the solver-state mutators (`add`, `remove`, `..._many`) and must be called
    with the same indices, in the same order.  Snapshot methods mirror the solver-state snapshot
    life cycle, which is a *stack*: `push_snapshot` saves the current contributions on top of any
    already saved, and `pop_snapshot` discards the top entry, restoring from it or not.  Numba
    kernels are only ever handed bare numpy arrays, never tracker objects.
    """

    # -------------------------------------------------------------------------
    #  Contribution reads
    # -------------------------------------------------------------------------
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

    @property
    @abstractmethod
    def contribution_wrt_dataset(self) -> NDArray[np.floating]:
        """Return static per-point contribution of each point wrt the whole dataset (selection-independent).

        Dataset-wide contributions live in a lazily filled cache: elements start as NaN (not yet
        computed), are computed on first read — each requiring a scan of one full row of the
        pairwise-distance matrix — and never change afterwards.  This property first computes
        every missing element, so it is always safe to read but costs the full O(n²) sweep on
        first access.  Callers needing only some elements use `contribution_wrt_dataset_for`,
        which computes just the elements it returns.

        The returned array is a reference to internal state — callers must not modify it.
        """
        raise NotImplementedError

    @abstractmethod
    def contribution_wrt_dataset_for(self, indices: NDArray[np.int32]) -> NDArray[np.float32]:
        """Return dataset-wide contributions for `indices`, computing missing entries first.

        The targeted counterpart of `contribution_wrt_dataset`: only the requested elements are
        guaranteed computed afterwards, so the cost is proportional to the not-yet-computed
        elements among `indices` rather than to n.

        Returns a freshly allocated array (safe for in-place mutation by the caller).
        """
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

    def add_many(self, indices: NDArray[np.int32], parallel: bool = False) -> None:
        """Update contributions after adding all points in `indices` to the selection.

        `parallel` lets the update run over parallel threads where a tracker implements that;
        results are identical either way, and this default implementation ignores the flag.
        Callers may only opt in when no other worker process is competing for the cores —
        inside one of several concurrently solving workers, the threads would oversubscribe the cores.
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
    #  Snapshot & copy
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

    @abstractmethod
    def copy(self) -> DiversityContributionTracker:
        """Return a deep copy of this tracker."""
        raise NotImplementedError
