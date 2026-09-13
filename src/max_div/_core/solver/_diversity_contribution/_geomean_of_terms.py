"""The tracker a hybrid objective exposes: the geometric mean, over its terms, of each term's tracker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from max_div._core._math.geomean import geomean_per_column_f32

from ._base import DiversityContributionTracker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceStore


# =================================================================================================
#  GeoMeanOfTermsTracker
# =================================================================================================
class GeoMeanOfTermsTracker(DiversityContributionTracker):
    """Per-point contribution of a hybrid objective: the geometric mean over its terms of each term's contribution.

    The tracker holds one member tracker per term of the objective, so a tracker that two terms
    read appears twice and counts twice, which keeps the combined value the item's exact per-item
    factor of a geometric-mean objective. It computes nothing incrementally: it reads its members
    and combines their arrays, and it is a member of the same tracker set as they are, so the
    set's mutation fan-out reaches it after reaching them. On a mutation it only marks its
    combined array stale; the members update themselves.

    It reads no single distance store, since its terms may read different ones; the store
    property raises, and the batched farthest-point initialization, the one reader of that
    store, never runs with a hybrid objective.
    """

    # -------------------------------------------------------------------------
    #  Construction & copy
    # -------------------------------------------------------------------------
    def __init__(self, term_trackers: Sequence[DiversityContributionTracker]) -> None:
        """Initialize over the members, one per term, in the order of the terms.

        Args:
            term_trackers: the tracker each term reads; at least two, and a tracker may appear more
                than once. A one-term geometric mean is the term's own tracker.

        Raises:
            ValueError: If fewer than two member trackers are given.
        """
        if len(term_trackers) < 2:
            raise ValueError(
                f"A geometric mean over terms needs at least two member trackers; got {len(term_trackers)}."
            )
        self._term_trackers = tuple(term_trackers)  # READ-ONLY
        n = self._term_trackers[0].store.n
        self._n_terms = len(self._term_trackers)
        # the members' arrays are stacked into this buffer, one row per term, before combining
        self._stacked = np.empty((self._n_terms, n), dtype=np.float32)
        # combined arrays: wrt the selection, stale after any mutation; wrt the dataset, computed once
        self._wrt_selection: NDArray[np.float32] | None = None
        self._wrt_dataset: NDArray[np.float32] | None = None

    def copy(self) -> GeoMeanOfTermsTracker:
        """Return a tracker over copies of the members.

        A tracker set that copies its own trackers builds the copy over those copies instead, so
        that the members stay the ones the set mutates.
        """
        return GeoMeanOfTermsTracker([tracker.copy() for tracker in self._term_trackers])

    @property
    def term_trackers(self) -> tuple[DiversityContributionTracker, ...]:
        """Return the member trackers, one per term."""
        return self._term_trackers

    # -------------------------------------------------------------------------
    #  Contribution reads
    # -------------------------------------------------------------------------
    @property
    def store(self) -> DistanceStore:
        """Raise: the terms may read different stores, so there is no single one to return."""
        raise ValueError("A geometric mean over terms reads no single distance store; read the members' stores.")

    def contribution_wrt_selection(self, selected: NDArray[np.bool], n_selected: np.int32) -> NDArray[np.float32]:
        """Return the geometric mean over terms of the members' selection contributions (reference; do not modify).

        Combined on the first read after a mutation and cached until the next one.
        """
        if self._wrt_selection is None:
            for row, tracker in enumerate(self._term_trackers):
                self._stacked[row, :] = tracker.contribution_wrt_selection(selected, n_selected)
            self._wrt_selection = np.empty(self._stacked.shape[1], dtype=np.float32)
            geomean_per_column_f32(self._stacked, self._wrt_selection)
        return self._wrt_selection

    @property
    def contribution_wrt_dataset(self) -> NDArray[np.float32]:
        """Return the geometric mean over terms of the members' dataset-wide contributions (reference; do not modify).

        Combined once, on first read, from the members' full arrays.
        """
        if self._wrt_dataset is None:
            for row, tracker in enumerate(self._term_trackers):
                self._stacked[row, :] = tracker.contribution_wrt_dataset
            self._wrt_dataset = np.empty(self._stacked.shape[1], dtype=np.float32)
            geomean_per_column_f32(self._stacked, self._wrt_dataset)
        return self._wrt_dataset

    def contribution_wrt_dataset_for(self, indices: NDArray[np.int32]) -> NDArray[np.float32]:
        """Return the combined dataset-wide contributions for `indices` only (fresh array)."""
        stacked = np.empty((self._n_terms, len(indices)), dtype=np.float32)
        for row, tracker in enumerate(self._term_trackers):
            stacked[row, :] = tracker.contribution_wrt_dataset_for(indices)
        combined = np.empty(len(indices), dtype=np.float32)
        geomean_per_column_f32(stacked, combined)
        return combined

    # -------------------------------------------------------------------------
    #  Mutations: the members update themselves; here only the combined array goes stale
    # -------------------------------------------------------------------------
    def add(self, index: np.int32) -> None:
        """Mark the combined selection contributions stale."""
        self._wrt_selection = None

    def add_many(self, indices: NDArray[np.int32], parallel: bool = False) -> None:
        """Mark the combined selection contributions stale."""
        self._wrt_selection = None

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Mark the combined selection contributions stale."""
        self._wrt_selection = None

    def remove_trial(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Mark the combined selection contributions stale."""
        self._wrt_selection = None

    def remove_many(self, indices: NDArray[np.int32], new_selection: NDArray[np.int32]) -> None:
        """Mark the combined selection contributions stale."""
        self._wrt_selection = None

    def reset(self) -> None:
        """Mark the combined selection contributions stale."""
        self._wrt_selection = None

    # -------------------------------------------------------------------------
    #  Snapshot
    # -------------------------------------------------------------------------
    def push_snapshot(self) -> None:
        """Do nothing: the members hold the snapshots, and the combined array is recomputed from them."""

    def pop_snapshot(self, restore: bool) -> None:
        """Mark the combined selection contributions stale when the members restore theirs."""
        if restore:
            self._wrt_selection = None
