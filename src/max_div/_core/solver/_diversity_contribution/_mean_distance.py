from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance

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
@numba.njit(numba.float64[::1](DISTANCE_STORE_TYPE), cache=True)
def compute_distance_sums(store: DistanceStore) -> NDArray[np.float64]:
    """Compute sum of distances of each item wrt all others, given the distance store."""
    n = store.n
    dist_sums = np.zeros(n, dtype=np.float64)
    # partners are visited in ascending index per item, so each float64 sum accumulates in the
    # same order however the store lays distances out — sums stay bit-equal across backends
    for i in np.arange(n, dtype=np.int32):
        for j in np.arange(n, dtype=np.int32):
            if j != i:
                dist_sums[i] += np.float64(get_distance(store, i, j))
    return dist_sums


@numba.njit(numba.void(numba.float64[::1], DISTANCE_STORE_TYPE, numba.int32), cache=True)
def update_distance_sums_add(dist_sums: NDArray[np.float64], store: DistanceStore, i_added: np.int32) -> None:
    """Update distance sums of each item wrt selection, given the distance store, after adding i_added."""
    for j in np.arange(store.n, dtype=np.int32):
        if j != i_added:
            dist_sums[j] += np.float64(get_distance(store, i_added, j))


@numba.njit(numba.void(numba.float64[::1], DISTANCE_STORE_TYPE, numba.int32), cache=True)
def update_distance_sums_remove(dist_sums: NDArray[np.float64], store: DistanceStore, i_removed: np.int32) -> None:
    """Update distance sums of each item wrt selection, given the distance store, after removing i_removed."""
    for j in np.arange(store.n, dtype=np.int32):
        if j != i_removed:
            dist_sums[j] -= np.float64(get_distance(store, i_removed, j))


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
        store: DistanceStore,
        contribution_wrt_dataset: NDArray[np.float32] | None = None,
        dist_sums: NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize the MeanDistanceTracker for an empty selection.

        :param store: (DistanceStore) pairwise-distance storage; immutable, so shareable across copies.
        :param contribution_wrt_dataset: (np.ndarray[np.float32] | None) precomputed global contribution;
                                    computed if omitted.
        :param dist_sums: (np.ndarray[np.float64] | None) current distance sums wrt selection; fresh (all 0.0,
                          i.e. empty selection) if omitted.  Together with `contribution_wrt_dataset` this
                          enables copies without recomputation.
        """
        self._store = store  # READ-ONLY
        if contribution_wrt_dataset is not None:
            self._contribution_wrt_dataset = contribution_wrt_dataset  # READ-ONLY
        else:
            n = int(store.n)
            self._contribution_wrt_dataset = (compute_distance_sums(store) / max(n - 1, 1)).astype(np.float32)
        self._dist_sums = dist_sums if dist_sums is not None else np.zeros(store.n, dtype=np.float64)
        # snapshot stack, innermost last; entries are owned copies handed back on a restoring pop
        self._snapshot_dist_sums: list[NDArray[np.float64]] = []

    def copy(self) -> MeanDistanceTracker:
        """Return an independent copy of this tracker; the immutable store and global contributions are shared."""
        return MeanDistanceTracker(
            store=self._store,
            contribution_wrt_dataset=self._contribution_wrt_dataset,
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
        update_distance_sums_add(self._dist_sums, self._store, index)

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update distance sums after removing point `index`.

        `new_selection` is not needed by this tracker: removal is exact subtraction.
        """
        update_distance_sums_remove(self._dist_sums, self._store, index)

    # -------------------------------------------------------------------------
    #  Snapshot
    # -------------------------------------------------------------------------
    def push_snapshot(self) -> None:
        """Save a copy of the current distance sums on top of the snapshot stack."""
        self._snapshot_dist_sums.append(self._dist_sums.copy())

    def pop_snapshot(self, restore: bool) -> None:
        """Discard the top snapshot, first restoring the distance sums from it if `restore`."""
        # no copy needed: the entry leaves the stack, so the restored array cannot alias a live snapshot
        snapshot = self._snapshot_dist_sums.pop()
        if restore:
            self._dist_sums = snapshot
