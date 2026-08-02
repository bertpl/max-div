from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance

from ._base import DiversityContributionTracker

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =================================================================================================
#  Separation kernels
# =================================================================================================
@numba.njit(numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32[::1]), cache=True)
def compute_separation_elements(sep: NDArray[np.float32], store: DistanceStore, indices: NDArray[np.int32]) -> None:
    """Fill the given elements of `sep` with each item's separation wrt all others.

    Each element requires scanning that item's full row of the pairwise-distance matrix.
    Elements are independent, so any subset can be computed in any order; a filled element
    equals what a full sweep would produce for it.
    """
    n = store.n
    for idx in indices:
        row_min = np.float32(np.inf)
        for j in np.arange(n, dtype=np.int32):
            if j != idx:
                dist_ij = get_distance(store, idx, j)
                if dist_ij < row_min:
                    row_min = dist_ij
        sep[idx] = row_min


@numba.njit(numba.float32[::1](DISTANCE_STORE_TYPE), cache=True)
def compute_separation(store: DistanceStore) -> NDArray[np.float32]:
    """Compute separation of each item wrt all others, given the distance store."""
    n = store.n
    sep = np.full(n, fill_value=np.inf, dtype=np.float32)
    compute_separation_elements(sep, store, np.arange(n, dtype=np.int32))
    return sep


@numba.njit(numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32), cache=True)
def update_separation_add(sep: NDArray[np.float32], store: DistanceStore, i_added: np.int32) -> None:
    """Update separation of each item wrt selection, given the distance store, after adding i_added."""
    for j in np.arange(store.n, dtype=np.int32):
        if j != i_added:
            dist = get_distance(store, i_added, j)
            if dist < sep[j]:
                sep[j] = dist


@numba.njit(numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32, numba.int32[::1]), cache=True)
def update_separation_remove(
    sep: NDArray[np.float32],
    store: DistanceStore,
    i_removed: np.int32,
    new_selection: NDArray[np.int32],
) -> None:
    """Update separation of each item wrt selection, given the distance store, after removing i_removed."""
    for j in np.arange(store.n, dtype=np.int32):
        if j != i_removed:
            dist = get_distance(store, i_removed, j)
            if dist <= sep[j]:
                # need to recompute sep[j]
                new_sep_j = np.inf
                for k in new_selection:
                    # only compute distance to currently selected items
                    if k != j:
                        dist_jk = get_distance(store, j, k)
                        if dist_jk < new_sep_j:
                            new_sep_j = dist_jk
                sep[j] = new_sep_j


# =================================================================================================
#  SeparationTracker
# =================================================================================================
class SeparationTracker(DiversityContributionTracker):
    """Diversity-contribution tracker of the separation family: contribution = distance to nearest selected point.

    For points with no selected neighbor (empty selection, or the point is the only selected one)
    the contribution is +inf.  The global contribution is each point's distance to its nearest neighbor in the
    whole dataset.
    """

    # -------------------------------------------------------------------------
    #  Construction & copy
    # -------------------------------------------------------------------------
    def __init__(
        self,
        store: DistanceStore,
        sep_global: NDArray[np.float32] | None = None,
        sep_selected: NDArray[np.float32] | None = None,
    ) -> None:
        """Initialize the SeparationTracker for an empty selection.

        :param store: (DistanceStore) pairwise-distance storage; immutable, so shareable across copies.
        :param sep_global: (np.ndarray[np.float32] | None) global-separation array to adopt; a fresh lazy
                           (all-NaN) array if omitted.
        :param sep_selected: (np.ndarray[np.float32] | None) current separations wrt selection; fresh (all +inf,
                             i.e. empty selection) if omitted.  Together with `sep_global` this enables copies
                             without recomputation.
        """
        self._store = store  # READ-ONLY
        # lazy memo: NaN = not yet computed; elements are filled on read and never change once
        # written (monotone), which is what makes sharing the array across copies safe
        self._sep_global = sep_global if sep_global is not None else np.full(store.n, np.nan, dtype=np.float32)
        self._sep_selected = sep_selected if sep_selected is not None else np.full(store.n, np.inf, dtype=np.float32)
        # snapshot stack, innermost last; entries are owned copies handed back on a restoring pop
        self._snapshot_sep_selected: list[NDArray[np.float32]] = []

    def copy(self) -> SeparationTracker:
        """Return an independent copy of this tracker; the store and the global-separation memo are shared.

        Sharing the memo is safe because its elements are monotone: filled on read, never
        rewritten, so copies can only ever benefit from each other's computed elements.
        """
        return SeparationTracker(
            store=self._store,
            sep_global=self._sep_global,
            sep_selected=self._sep_selected.copy(),
        )

    # -------------------------------------------------------------------------
    #  Contribution reads
    # -------------------------------------------------------------------------
    def contribution_wrt_selection(self, selected: NDArray[np.bool], n_selected: np.int32) -> NDArray[np.float32]:
        """Return separation of all points wrt the current selection (reference; do not modify)."""
        return self._sep_selected

    @property
    def contribution_wrt_dataset(self) -> NDArray[np.float32]:
        """Return separation of all points wrt all other points (reference; do not modify).

        Computes every not-yet-computed element first; see the base class for the lazy contract.
        """
        self._ensure_global_elements(np.arange(self._store.n, dtype=np.int32))
        return self._sep_global

    def contribution_wrt_dataset_for(self, indices: NDArray[np.int32]) -> NDArray[np.float32]:
        """Return global separations for `indices`, computing missing elements first (fresh array)."""
        self._ensure_global_elements(indices)
        return self._sep_global[indices]

    def _ensure_global_elements(self, indices: NDArray[np.int32]) -> None:
        """Compute any not-yet-computed global-separation elements among `indices`."""
        missing = indices[np.isnan(self._sep_global[indices])]
        if missing.size > 0:
            compute_separation_elements(self._sep_global, self._store, np.ascontiguousarray(missing, dtype=np.int32))

    # -------------------------------------------------------------------------
    #  Mutations
    # -------------------------------------------------------------------------
    def add(self, index: np.int32) -> None:
        """Update separations after adding point `index` to the selection."""
        update_separation_add(self._sep_selected, self._store, index)

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update separations after removing point `index`, rescanning against `new_selection` where needed."""
        update_separation_remove(self._sep_selected, self._store, index, new_selection)

    # -------------------------------------------------------------------------
    #  Snapshot
    # -------------------------------------------------------------------------
    def push_snapshot(self) -> None:
        """Save a copy of the current separations on top of the snapshot stack."""
        self._snapshot_sep_selected.append(self._sep_selected.copy())

    def pop_snapshot(self, restore: bool) -> None:
        """Discard the top snapshot, first restoring the separations from it if `restore`."""
        # no copy needed: the entry leaves the stack, so the restored array cannot alias a live snapshot
        snapshot = self._snapshot_sep_selected.pop()
        if restore:
            self._sep_selected = snapshot
