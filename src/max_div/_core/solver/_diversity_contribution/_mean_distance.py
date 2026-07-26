from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import get_pdist_el

from ._base import DiversityContributionTracker

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =================================================================================================
#  Distance sums
# =================================================================================================
# Per-point sums of distances to a selection — the pairwise-distance counterpart of the separation
# kernels above.  Sums accumulate in float64: entries undergo long add/subtract chains over solver
# iterations, and float32 drift there would change scores with iteration count.  Distances of a
# point to itself are 0, so a point's own entry never needs special-casing: it always equals the
# sum of its distances to the *other* selected points.
@numba.njit("float64[::1](float32[::1], int32)", cache=True)
def compute_distance_sums(pdist: NDArray[np.float32], n: np.int32) -> NDArray[np.float64]:
    """Compute sum of distances of each item wrt all others, given pairwise distance array pdist and n items."""
    dist_sums = np.zeros(n, dtype=np.float64)
    pdist_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # note: the way we iterate over i & j represents the exact order in which pdist stores distances
            dist_ij = np.float64(pdist[pdist_idx])
            pdist_idx += 1
            dist_sums[i] += dist_ij
            dist_sums[j] += dist_ij
    return dist_sums


@numba.njit("void(float64[::1], float32[::1], int32, int32)", cache=True)
def update_distance_sums_add(
    dist_sums: NDArray[np.float64], pdist: NDArray[np.float32], n: np.int32, i_added: np.int32
) -> None:
    """Update distance sums of each item wrt selection, given pdist array and n items, after adding i_added."""
    for j in np.arange(n, dtype=np.int32):
        if j != i_added:
            dist_sums[j] += np.float64(get_pdist_el(pdist, i_added, j, n))


@numba.njit("void(float64[::1], float32[::1], int32, int32)", cache=True)
def update_distance_sums_remove(
    dist_sums: NDArray[np.float64], pdist: NDArray[np.float32], n: np.int32, i_removed: np.int32
) -> None:
    """Update distance sums of each item wrt selection, given pdist array and n items, after removing i_removed."""
    for j in np.arange(n, dtype=np.int32):
        if j != i_removed:
            dist_sums[j] -= np.float64(get_pdist_el(pdist, i_removed, j, n))


# =================================================================================================
#  MeanDistanceTracker
# =================================================================================================
class MeanDistanceTracker(DiversityContributionTracker):
    """Diversity-contribution tracker of the mean-distance family: contribution = mean distance to selected points.

    Internally tracks *raw sums* of distances in float64 — incremental updates stay exact add/subtract
    arithmetic, free of the rescaling (and rounding) that maintaining means directly would need.  Contribution
    reads expose *mean form* (sum / number of selected neighbors), so values stay in the same "an average
    distance" unit as other contribution families, and are returned as float32 like all exposed contribution arrays.

    The number of selected neighbors is membership-aware: a selected point's own zero self-distance is
    not a neighbor, so its divisor is one less than a non-selected point's.  For points with no selected
    neighbor the contribution is 0.  The global contribution is each point's mean distance to all other points.
    """

    # -------------------------------------------------------------------------
    #  Construction & copy
    # -------------------------------------------------------------------------
    def __init__(
        self,
        pdist: NDArray[np.float32],
        n: np.int32,
        contribution_wrt_dataset: NDArray[np.float32] | None = None,
        dist_sums: NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize the MeanDistanceTracker for an empty selection.

        :param pdist: (np.ndarray[np.float32]) condensed pair-wise distance vector (1D array of size (n*(n-1))//2)
        :param n: (np.int32) number of items
        :param contribution_wrt_dataset: (np.ndarray[np.float32] | None) precomputed global contribution;
                                    computed if omitted.
        :param dist_sums: (np.ndarray[np.float64] | None) current distance sums wrt selection; fresh (all 0.0,
                          i.e. empty selection) if omitted.  Together with `contribution_wrt_dataset` this
                          enables copies without recomputation.
        """
        self._pdist = pdist  # READ-ONLY
        self._n = n  # READ-ONLY
        if contribution_wrt_dataset is not None:
            self._contribution_wrt_dataset = contribution_wrt_dataset  # READ-ONLY
        else:
            self._contribution_wrt_dataset = (compute_distance_sums(pdist, n) / max(int(n) - 1, 1)).astype(np.float32)
        self._dist_sums = dist_sums if dist_sums is not None else np.zeros(n, dtype=np.float64)
        self._snapshot_dist_sums: NDArray[np.float64] = _EMPTY_NP_ARRAY_FLOAT64

    def copy(self) -> MeanDistanceTracker:
        """Return a deep copy of this tracker (without recomputing the global contribution)."""
        return MeanDistanceTracker(
            pdist=self._pdist.copy(),
            n=self._n,
            contribution_wrt_dataset=self._contribution_wrt_dataset.copy(),
            dist_sums=self._dist_sums.copy(),
        )

    # -------------------------------------------------------------------------
    #  Contribution reads
    # -------------------------------------------------------------------------
    def contribution_wrt_selection(self, selected: NDArray[np.bool], n_selected: np.int32) -> NDArray[np.float32]:
        """Return mean distance of all points wrt the current selection (freshly allocated array)."""
        # per-point divisor: number of selected neighbors — a selected point's own 0-distance is not a neighbor
        divisor = np.maximum(n_selected - selected, 1)  # bool subtraction; clip avoids 0/0 for empty neighborhoods
        return (self._dist_sums / divisor).astype(np.float32)

    @property
    def contribution_wrt_dataset(self) -> NDArray[np.float32]:
        """Return mean distance of all points wrt all other points (reference; do not modify)."""
        return self._contribution_wrt_dataset

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
