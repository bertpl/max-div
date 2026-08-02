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
@numba.njit(numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32[::1]), cache=True)
def compute_mean_distance_elements(out: NDArray[np.float32], store: DistanceStore, indices: NDArray[np.int32]) -> None:
    """Fill the given elements of `out` with each item's mean distance to all others.

    Each element requires scanning that item's full row of the pairwise-distance matrix; elements
    are independent, so any subset can be computed in any order.  Per element: a float64 sum over
    partners in ascending index (so sums stay bit-equal across store layouts), one divide by
    (n-1), one cast to float32.
    """
    n = store.n
    den = np.float64(max(n - 1, 1))
    for idx in indices:
        row_sum = np.float64(0.0)
        for j in np.arange(n, dtype=np.int32):
            if j != idx:
                row_sum += np.float64(get_distance(store, idx, j))
        out[idx] = np.float32(row_sum / den)


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
        :param contribution_wrt_dataset: (np.ndarray[np.float32] | None) global-contribution array to adopt;
                                    a fresh lazy (all-NaN) array if omitted.
        :param dist_sums: (np.ndarray[np.float64] | None) current distance sums wrt selection; fresh (all 0.0,
                          i.e. empty selection) if omitted.  Together with `contribution_wrt_dataset` this
                          enables copies without recomputation.
        """
        self._store = store  # READ-ONLY
        if contribution_wrt_dataset is not None:
            self._contribution_wrt_dataset = contribution_wrt_dataset
        else:
            # lazily filled cache: NaN marks a not-yet-computed element; elements are computed on
            # read and never change afterwards, which is what makes sharing the array across copies safe
            self._contribution_wrt_dataset = np.full(store.n, np.nan, dtype=np.float32)
        self._dist_sums = dist_sums if dist_sums is not None else np.zeros(store.n, dtype=np.float64)
        # snapshot stack, innermost last; entries are owned copies handed back on a restoring pop
        self._snapshot_dist_sums: list[NDArray[np.float64]] = []

    def copy(self) -> MeanDistanceTracker:
        """Return an independent copy of this tracker; store and lazily filled cache are shared.

        Sharing the global-contribution cache is safe because its elements are computed once and
        never rewritten, so copies can only ever benefit from each other's computed elements.
        """
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
        """Return mean distance of all points wrt all other points (reference; do not modify).

        Computes every not-yet-computed element first; see the base class for the lazy contract.
        """
        self._ensure_global_elements(np.arange(self._store.n, dtype=np.int32))
        return self._contribution_wrt_dataset

    def contribution_wrt_dataset_for(self, indices: NDArray[np.int32]) -> NDArray[np.float32]:
        """Return global mean-distance contributions for `indices`, computing missing elements first (fresh array)."""
        self._ensure_global_elements(indices)
        return self._contribution_wrt_dataset[indices]

    def _ensure_global_elements(self, indices: NDArray[np.int32]) -> None:
        """Compute any not-yet-computed global-contribution elements among `indices`."""
        missing = indices[np.isnan(self._contribution_wrt_dataset[indices])]
        if missing.size > 0:
            compute_mean_distance_elements(
                self._contribution_wrt_dataset, self._store, np.ascontiguousarray(missing, dtype=np.int32)
            )

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
