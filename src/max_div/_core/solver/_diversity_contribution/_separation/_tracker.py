"""The separation tracker: contribution = distance to the nearest selected item."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._base import DiversityContributionTracker
from ._backends import backend_for

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceStore


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
        # the layout is a property of the store, so which calculations apply is settled here
        # rather than tested inside them; see `_backends` for why that test cannot live in
        # compiled code
        self._backend = backend_for(store)
        # lazily filled cache: NaN marks a not-yet-computed element; elements are computed on read
        # and never change afterwards, which is what makes sharing the array across copies safe
        self._sep_global = sep_global if sep_global is not None else np.full(store.n, np.nan, dtype=np.float32)
        self._sep_selected = sep_selected if sep_selected is not None else np.full(store.n, np.inf, dtype=np.float32)
        # snapshot stack, innermost last; entries are owned copies handed back on a restoring pop
        self._snapshot_sep_selected: list[NDArray[np.float32]] = []

    def copy(self) -> SeparationTracker:
        """Return an independent copy of this tracker; store and lazily filled cache are shared.

        Sharing the global-separation cache is safe because its elements are computed once and
        never rewritten, so copies can only ever benefit from each other's computed elements.
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
            self._backend.elements(self._sep_global, self._store, np.ascontiguousarray(missing, dtype=np.int32))

    # -------------------------------------------------------------------------
    #  Mutations
    # -------------------------------------------------------------------------
    def add(self, index: np.int32) -> None:
        """Update separations after adding point `index` to the selection."""
        self._backend.add(self._sep_selected, self._store, index)

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update separations after removing point `index`, rescanning against `new_selection` where needed."""
        self._backend.remove(self._sep_selected, self._store, index, new_selection)

    def reset(self) -> None:
        """Reset separations to the empty selection (all +inf); the global cache stays valid as-is."""
        self._sep_selected.fill(np.inf)

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
