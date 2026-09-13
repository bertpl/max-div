"""The contribution tracker of a `DiversityObjectiveHybridGeoMean`; see `HybridGeoMeanTracker`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from max_div._core._math.geomean import geomean_per_row_f32

from ._base import DiversityContributionTracker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceStore


# =================================================================================================
#  HybridGeoMeanTracker
# =================================================================================================
class HybridGeoMeanTracker(DiversityContributionTracker):
    """Per-point contribution of a geometric-mean hybrid objective: the geometric mean of the per-term contributions.

    The tracker holds one term tracker per term of the objective, in term order. Two terms may read
    the same tracker; then that tracker appears twice in the list and enters the mean twice, as the
    objective itself counts a repeated term twice.

    It keeps no incremental state of its own: it reads the term trackers and combines their arrays.
    Its mutation methods only mark the combined array stale and do not forward the mutation, because
    the term trackers are also the trackers a tracker set holds by spec and updates itself, so
    forwarding would apply every mutation twice. They must therefore be called after the term
    trackers' own mutation methods, and `pop_snapshot(restore=True)` marks the combined array stale
    the same way.
    """

    # -------------------------------------------------------------------------
    #  Construction
    # -------------------------------------------------------------------------
    def __init__(self, term_trackers: Sequence[DiversityContributionTracker]) -> None:
        """Initialize from the term trackers, one per term, in the order of the terms.

        Args:
            term_trackers: One tracker per term, in term order; at least two. The same tracker may
                appear more than once.

        Raises:
            ValueError: If fewer than two term trackers are given.
        """
        if len(term_trackers) < 2:
            raise ValueError(f"A hybrid geometric mean needs at least two term trackers; got {len(term_trackers)}.")
        self._term_trackers = tuple(term_trackers)  # READ-ONLY
        self._contribution_wrt_selection: NDArray[np.float32] | None = None
        self._contribution_wrt_dataset: NDArray[np.float32] | None = None

    @property
    def term_trackers(self) -> tuple[DiversityContributionTracker, ...]:
        """Return the term trackers, one per term."""
        return self._term_trackers

    # -------------------------------------------------------------------------
    #  Contribution reads
    # -------------------------------------------------------------------------
    @property
    def store(self) -> DistanceStore:
        """Raise: the terms may read different stores, so there is no single one to return."""
        raise ValueError("A hybrid geometric mean reads no single distance store; read the term trackers' stores.")

    def contribution_wrt_selection(self, selected: NDArray[np.bool], n_selected: np.int32) -> NDArray[np.float32]:
        """Return the geometric mean of the per-term selection contributions (reference; do not modify).

        Combined on the first read after a mutation and cached until the next mutation.
        """
        if self._contribution_wrt_selection is None:
            self._contribution_wrt_selection = self._geomean_over_terms(
                [tracker.contribution_wrt_selection(selected, n_selected) for tracker in self._term_trackers]
            )
        return self._contribution_wrt_selection

    @property
    def contribution_wrt_dataset(self) -> NDArray[np.float32]:
        """Return the geometric mean of the per-term dataset-wide contributions (reference; do not modify).

        Combined once, on first read, from the term trackers' full arrays.
        """
        if self._contribution_wrt_dataset is None:
            self._contribution_wrt_dataset = self._geomean_over_terms(
                [tracker.contribution_wrt_dataset for tracker in self._term_trackers]
            )
        return self._contribution_wrt_dataset

    def contribution_wrt_dataset_for(self, indices: NDArray[np.int32]) -> NDArray[np.float32]:
        """Return the combined dataset-wide contributions for `indices` only (fresh array)."""
        return self._geomean_over_terms(
            [tracker.contribution_wrt_dataset_for(indices) for tracker in self._term_trackers]
        )

    @staticmethod
    def _geomean_over_terms(term_contributions: list[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Return the elementwise geometric mean of the per-term arrays, as a fresh float32 array."""
        # one row per point, one column per term, so each point's values are contiguous
        stacked = np.stack(term_contributions, axis=1).astype(np.float32, copy=False)
        geomean_contributions = np.empty(stacked.shape[0], dtype=np.float32)
        geomean_per_row_f32(stacked, geomean_contributions)
        return geomean_contributions

    # -------------------------------------------------------------------------
    #  Mutations: the term trackers update themselves; here only the combined array goes stale
    # -------------------------------------------------------------------------
    def _mark_selection_contribution_stale(self) -> None:
        """Drop the cached combined selection contributions, so the next read recombines the term trackers."""
        self._contribution_wrt_selection = None

    def add(self, index: np.int32) -> None:
        """Mark the combined selection contributions stale."""
        self._mark_selection_contribution_stale()

    def add_many(self, indices: NDArray[np.int32], parallel: bool = False) -> None:
        """Mark the combined selection contributions stale."""
        self._mark_selection_contribution_stale()

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Mark the combined selection contributions stale."""
        self._mark_selection_contribution_stale()

    def remove_trial(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Mark the combined selection contributions stale."""
        self._mark_selection_contribution_stale()

    def remove_many(self, indices: NDArray[np.int32], new_selection: NDArray[np.int32]) -> None:
        """Mark the combined selection contributions stale."""
        self._mark_selection_contribution_stale()

    def reset(self) -> None:
        """Mark the combined selection contributions stale."""
        self._mark_selection_contribution_stale()

    # -------------------------------------------------------------------------
    #  Snapshot
    # -------------------------------------------------------------------------
    def push_snapshot(self) -> None:
        """Do nothing: the term trackers hold the snapshots, and the combined array is recomputed from them."""

    def pop_snapshot(self, restore: bool) -> None:
        """Mark the combined selection contributions stale when `restore` is true.

        The term trackers then restore their own arrays, so the combined one no longer matches them.
        """
        if restore:
            self._mark_selection_contribution_stale()
