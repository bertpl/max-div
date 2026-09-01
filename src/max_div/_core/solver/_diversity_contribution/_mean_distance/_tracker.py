"""The mean-distance tracker: contribution = mean distance to the selected items."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._base import DiversityContributionTracker
from ._backends import backend_for

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceStore


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

        Args:
            store: (DistanceStore) pairwise-distance storage; immutable, so shareable across copies.
            contribution_wrt_dataset: (np.ndarray[np.float32] | None) global-contribution array to adopt;
                a fresh lazy (all-NaN) array if omitted.
            dist_sums: (np.ndarray[np.float64] | None) current distance sums wrt selection; fresh (all 0.0,
                i.e. empty selection) if omitted.  Together with `contribution_wrt_dataset` this
                enables copies without recomputation.
        """
        self._store = store  # READ-ONLY
        # the layout is a property of the store, so which calculations apply is settled here
        # rather than tested inside them; see `_backends` for why that test cannot live in
        # compiled code
        self._backend = backend_for(store)
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

    @property
    def store(self) -> DistanceStore:
        """Return the distance store this tracker reads (shared, immutable)."""
        return self._store

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
            self._backend.elements(
                self._contribution_wrt_dataset, self._store, np.ascontiguousarray(missing, dtype=np.int32)
            )

    # -------------------------------------------------------------------------
    #  Mutations
    # -------------------------------------------------------------------------
    def add(self, index: np.int32) -> None:
        """Update distance sums after adding point `index` to the selection."""
        self._backend.add(self._dist_sums, self._store, index)

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update distance sums after removing point `index`.

        `new_selection` is not needed by this tracker: removal is exact subtraction.
        """
        self._backend.remove(self._dist_sums, self._store, index)

    def reset(self) -> None:
        """Reset distance sums to the empty selection (all zero); the global cache stays valid as-is."""
        self._dist_sums.fill(0.0)

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
