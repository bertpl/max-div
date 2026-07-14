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
    life cycle: `set_snapshot` overwrites any previous snapshot; `restore_snapshot` restores and
    invalidates it.  Numba kernels are only ever handed bare numpy arrays, never tracker objects.
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

        :param selected: (n-sized bool ndarray) current selection mask.
        :param n_selected: (np.int32) number of True values in `selected`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def contribution_wrt_dataset(self) -> NDArray[np.floating]:
        """Return static per-point contribution of each point wrt the whole dataset (selection-independent).

        The returned array is a reference to internal state — callers must not modify it.
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

        :param index: (np.int32) the point just removed.
        :param new_selection: (int32 ndarray) indices selected *after* the removal — needed by
                              trackers whose contribution requires rescanning the remaining selection.
        """
        raise NotImplementedError

    def add_many(self, indices: NDArray[np.int32]) -> None:
        """Update contributions after adding all points in `indices` to the selection."""
        for index in indices:
            self.add(index)

    def remove_many(self, indices: NDArray[np.int32], new_selection: NDArray[np.int32]) -> None:
        """Update contributions after removing all points in `indices` from the selection.

        :param indices: (int32 ndarray) the points just removed.
        :param new_selection: (int32 ndarray) indices selected after *all* removals.
        """
        for index in indices:
            self.remove(index, new_selection)

    # -------------------------------------------------------------------------
    #  Snapshot & copy
    # -------------------------------------------------------------------------
    @abstractmethod
    def set_snapshot(self) -> None:
        """Internally save the current contribution state, overwriting any previous snapshot."""
        raise NotImplementedError

    @abstractmethod
    def restore_snapshot(self) -> None:
        """Restore the contribution state saved by the last `set_snapshot` call and invalidate it."""
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> DiversityContributionTracker:
        """Return a deep copy of this tracker."""
        raise NotImplementedError
