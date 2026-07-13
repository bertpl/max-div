from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from max_div._core.metrics._distance import (
    compute_distance_sums,
    update_distance_sums_add,
    update_distance_sums_remove,
)

from ._base import DiversitySignalTracker

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =================================================================================================
#  MeanDistanceTracker
# =================================================================================================
class MeanDistanceTracker(DiversitySignalTracker):
    """Diversity-signal tracker of the mean-distance family: signal = mean distance to selected points.

    Internally tracks *raw sums* of distances in float64 — incremental updates stay exact add/subtract
    arithmetic, free of the rescaling (and rounding) that maintaining means directly would need.  Signal
    reads expose *mean form* (sum / number of selected neighbors), so values stay in the same "an average
    distance" unit as other signal families, and are returned as float32 like all exposed signal arrays.

    The number of selected neighbors is membership-aware: a selected point's own zero self-distance is
    not a neighbor, so its divisor is one less than a non-selected point's.  For points with no selected
    neighbor the signal is 0.  The global signal is each point's mean distance to all other points.
    """

    # -------------------------------------------------------------------------
    #  Construction & copy
    # -------------------------------------------------------------------------
    def __init__(
        self,
        pdist: NDArray[np.float32],
        n: np.int32,
        global_signal: NDArray[np.float32] | None = None,
        dist_sums: NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize the MeanDistanceTracker for an empty selection.

        :param pdist: (np.ndarray[np.float32]) condensed pair-wise distance vector (1D array of size (n*(n-1))//2)
        :param n: (np.int32) number of vectors
        :param global_signal: (np.ndarray[np.float32] | None) precomputed global signal; computed if omitted.
        :param dist_sums: (np.ndarray[np.float64] | None) current distance sums wrt selection; fresh (all 0.0,
                          i.e. empty selection) if omitted.  Together with `global_signal` this enables copies
                          without recomputation.
        """
        self._pdist = pdist  # READ-ONLY
        self._n = n  # READ-ONLY
        if global_signal is not None:
            self._global_signal = global_signal  # READ-ONLY
        else:
            self._global_signal = (compute_distance_sums(pdist, n) / max(int(n) - 1, 1)).astype(np.float32)
        self._dist_sums = dist_sums if dist_sums is not None else np.zeros(n, dtype=np.float64)
        self._snapshot_dist_sums: NDArray[np.float64] = _EMPTY_NP_ARRAY_FLOAT64

    def copy(self) -> MeanDistanceTracker:
        """Return a deep copy of this tracker (without recomputing the global signal)."""
        return MeanDistanceTracker(
            pdist=self._pdist.copy(),
            n=self._n,
            global_signal=self._global_signal.copy(),
            dist_sums=self._dist_sums.copy(),
        )

    # -------------------------------------------------------------------------
    #  Signal reads
    # -------------------------------------------------------------------------
    def full_signal(self, selected: NDArray[np.bool], n_selected: np.int32) -> NDArray[np.float32]:
        """Return mean distance of all points wrt the current selection (freshly allocated array)."""
        # per-point divisor: number of selected neighbors — a selected point's own 0-distance is not a neighbor
        divisor = np.maximum(n_selected - selected, 1)  # bool subtraction; clip avoids 0/0 for empty neighborhoods
        return (self._dist_sums / divisor).astype(np.float32)

    @property
    def global_signal(self) -> NDArray[np.float32]:
        """Return mean distance of all points wrt all other points (reference; do not modify)."""
        return self._global_signal

    # -------------------------------------------------------------------------
    #  Mutations
    # -------------------------------------------------------------------------
    def add(self, index: np.int32) -> None:
        """Update distance sums after adding point `index` to the selection."""
        update_distance_sums_add(self._dist_sums, self._pdist, self._n, index)

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update distance sums after removing point `index`.

        `new_selection` is not needed by this tracker: removal is exact subtraction.
        """
        update_distance_sums_remove(self._dist_sums, self._pdist, self._n, index)

    # -------------------------------------------------------------------------
    #  Snapshot
    # -------------------------------------------------------------------------
    def set_snapshot(self) -> None:
        """Save a copy of the current distance sums, overwriting any previous snapshot."""
        self._snapshot_dist_sums = self._dist_sums.copy()

    def restore_snapshot(self) -> None:
        """Restore the distance sums saved by the last `set_snapshot` call and invalidate the snapshot."""
        # no copy needed: the snapshot slot is cleared, so the restored array cannot alias a live snapshot
        self._dist_sums = self._snapshot_dist_sums
        self._snapshot_dist_sums = _EMPTY_NP_ARRAY_FLOAT64


# singleton to avoid repeated, unnecessary allocations for the invalid-snapshot placeholder
_EMPTY_NP_ARRAY_FLOAT64 = np.array([], dtype=np.float64)
