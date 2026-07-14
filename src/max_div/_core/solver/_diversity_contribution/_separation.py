from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from max_div._core.metrics._distance import get_pdist_el

from ._base import DiversityContributionTracker

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =================================================================================================
#  Separation kernels
# =================================================================================================
@numba.njit("float32[::1](float32[::1], int32)", cache=True)
def compute_separation(pdist: NDArray[np.float32], n: np.int32) -> NDArray[np.float32]:
    """Compute separation of each vector wrt all others, given pairwise distance array pdist and n vectors in total."""
    sep = np.full(n, fill_value=np.inf, dtype=np.float32)
    pdist_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # note: the way we iterate over i & j represents the exact order in which pdist stores distances
            dist_ij = pdist[pdist_idx]
            pdist_idx += 1
            sep[i] = min(sep[i], dist_ij)
            sep[j] = min(sep[j], dist_ij)
    return sep


@numba.njit("void(float32[::1], float32[::1], int32, int32)", cache=True)
def update_separation_add(sep: NDArray[np.float32], pdist: NDArray[np.float32], n: np.int32, i_added: np.int32) -> None:
    """Update separation of each vector wrt selection, given pdist array and n vectors, after adding i_added."""
    for j in np.arange(n, dtype=np.int32):
        if j != i_added:
            dist = get_pdist_el(pdist, i_added, j, n)
            if dist < sep[j]:
                sep[j] = dist


@numba.njit("void(float32[::1], float32[::1], int32, int32, int32[::1])", cache=True)
def update_separation_remove(
    sep: NDArray[np.float32],
    pdist: NDArray[np.float32],
    n: np.int32,
    i_removed: np.int32,
    new_selection: NDArray[np.int32],
) -> None:
    """Update separation of each vector wrt selection, given pdist array and n vectors, after removing i_removed."""
    for j in np.arange(n, dtype=np.int32):
        if j != i_removed:
            dist = get_pdist_el(pdist, i_removed, j, n)
            if dist <= sep[j]:
                # need to recompute sep[j]
                new_sep_j = np.inf
                for k in new_selection:
                    # only compute distance to currently selected vectors
                    if k != j:
                        dist_jk = get_pdist_el(pdist, j, k, n)
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
        pdist: NDArray[np.float32],
        n: np.int32,
        sep_global: NDArray[np.float32] | None = None,
        sep_selected: NDArray[np.float32] | None = None,
    ) -> None:
        """Initialize the SeparationTracker for an empty selection.

        :param pdist: (np.ndarray[np.float32]) condensed pair-wise distance vector (1D array of size (n*(n-1))//2)
        :param n: (np.int32) number of vectors
        :param sep_global: (np.ndarray[np.float32] | None) precomputed global separations; computed if omitted.
        :param sep_selected: (np.ndarray[np.float32] | None) current separations wrt selection; fresh (all +inf,
                             i.e. empty selection) if omitted.  Together with `sep_global` this enables copies
                             without recomputation.
        """
        self._pdist = pdist  # READ-ONLY
        self._n = n  # READ-ONLY
        self._sep_global = sep_global if sep_global is not None else compute_separation(pdist, n)  # READ-ONLY
        self._sep_selected = sep_selected if sep_selected is not None else np.full(n, np.inf, dtype=np.float32)
        self._snapshot_sep_selected: NDArray[np.float32] = _EMPTY_NP_ARRAY_FLOAT32

    def copy(self) -> SeparationTracker:
        """Return a deep copy of this tracker (without recomputing global separations)."""
        return SeparationTracker(
            pdist=self._pdist.copy(),
            n=self._n,
            sep_global=self._sep_global.copy(),
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
        """Return separation of all points wrt all other points (reference; do not modify)."""
        return self._sep_global

    # -------------------------------------------------------------------------
    #  Mutations
    # -------------------------------------------------------------------------
    def add(self, index: np.int32) -> None:
        """Update separations after adding point `index` to the selection."""
        update_separation_add(self._sep_selected, self._pdist, self._n, index)

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update separations after removing point `index`, rescanning against `new_selection` where needed."""
        update_separation_remove(self._sep_selected, self._pdist, self._n, index, new_selection)

    # -------------------------------------------------------------------------
    #  Snapshot
    # -------------------------------------------------------------------------
    def set_snapshot(self) -> None:
        """Save a copy of the current separations, overwriting any previous snapshot."""
        self._snapshot_sep_selected = self._sep_selected.copy()

    def restore_snapshot(self) -> None:
        """Restore the separations saved by the last `set_snapshot` call and invalidate the snapshot."""
        # no copy needed: the snapshot slot is cleared, so the restored array cannot alias a live snapshot
        self._sep_selected = self._snapshot_sep_selected
        self._snapshot_sep_selected = _EMPTY_NP_ARRAY_FLOAT32


# singleton to avoid repeated, unnecessary allocations for the invalid-snapshot placeholder
_EMPTY_NP_ARRAY_FLOAT32 = np.array([], dtype=np.float32)
