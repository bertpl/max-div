"""The per-item contribution source of an objective over several specs; see `CombinedPerItemContributionSource`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import PerItemContributionSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.metrics import DiversityObjective

    from ._base import DiversityContributionTracker


# =================================================================================================
#  CombinedPerItemContributionSource
# =================================================================================================
class CombinedPerItemContributionSource(PerItemContributionSource):
    """A combined source provides an objective's per-item contribution over several specs from its term trackers.

    Every read combines the term trackers' arrays by the objective's own rule
    (`compute_per_item_contributions`). The source caches no selection contribution: the tracker set
    maintains the term trackers on every selection change, and the strategies read the combined array
    once per change, so a cache would never serve a second read. The dataset-wide array never changes,
    so it is combined once.
    """

    # -------------------------------------------------------------------------
    #  Construction
    # -------------------------------------------------------------------------
    def __init__(self, objective: DiversityObjective, term_trackers: Sequence[DiversityContributionTracker]) -> None:
        """Bind the objective's combination to its term trackers.

        Args:
            objective: the objective whose per-item contribution this source provides.
            term_trackers: one tracker per entry of the objective's `tracker_specs`, in that order;
                the same tracker may appear more than once.

        Raises:
            ValueError: If fewer than two term trackers are given; a single spec's tracker is its own
                source.
        """
        if len(term_trackers) < 2:
            raise ValueError(f"A combined source needs at least two term trackers; got {len(term_trackers)}.")
        self._objective = objective  # READ-ONLY
        self._term_trackers = tuple(term_trackers)  # READ-ONLY
        self._contribution_wrt_dataset: NDArray[np.float32] | None = None

    @property
    def term_trackers(self) -> tuple[DiversityContributionTracker, ...]:
        """Return the term trackers, one per spec of the objective."""
        return self._term_trackers

    # -------------------------------------------------------------------------
    #  Contribution reads
    # -------------------------------------------------------------------------
    def contribution_wrt_selection(self, selected: NDArray[np.bool], n_selected: np.int32) -> NDArray[np.float32]:
        """Return the objective's combination of the term trackers' selection contributions (fresh array)."""
        return self._objective.compute_per_item_contributions(
            [tracker.contribution_wrt_selection(selected, n_selected) for tracker in self._term_trackers]
        )

    @property
    def contribution_wrt_dataset(self) -> NDArray[np.float32]:
        """Return the aggregation of the term trackers' dataset-wide contributions (reference; do not modify)."""
        if self._contribution_wrt_dataset is None:
            self._contribution_wrt_dataset = self._objective.compute_per_item_contributions(
                [tracker.contribution_wrt_dataset for tracker in self._term_trackers]
            )
        return self._contribution_wrt_dataset

    def contribution_wrt_dataset_for(self, indices: NDArray[np.int32]) -> NDArray[np.float32]:
        """Return the combined dataset-wide contributions for `indices` only (fresh array)."""
        return self._objective.compute_per_item_contributions(
            [tracker.contribution_wrt_dataset_for(indices) for tracker in self._term_trackers]
        )
