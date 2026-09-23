"""`HybridPerItemContributionSource` is the per-item contribution source of a hybrid objective."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import PerItemContributionSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.metrics import DiversityObjectiveHybrid

    from ._base import DiversityContributionTracker


# =================================================================================================
#  HybridPerItemContributionSource
# =================================================================================================
class HybridPerItemContributionSource(PerItemContributionSource):
    """A hybrid source provides a hybrid objective's per-item contribution from its term trackers, one per term.

    Every read combines the term trackers' arrays by the objective's own rule
    (`compute_per_item_contributions`). The source caches no selection contribution: the tracker set
    maintains the term trackers on every selection change, and the strategies read the combined array
    once per change, so a cache would never be read a second time.
    """

    # -------------------------------------------------------------------------
    #  Construction
    # -------------------------------------------------------------------------
    def __init__(
        self, objective: DiversityObjectiveHybrid, term_trackers: Sequence[DiversityContributionTracker]
    ) -> None:
        """Bind the objective's combination to its term trackers.

        Args:
            objective: the hybrid objective whose per-item contribution this source provides.
            term_trackers: one tracker per entry of the objective's `tracker_specs`, in that order;
                the same tracker may appear more than once.

        Raises:
            ValueError: If fewer than two term trackers are given; a hybrid has at least two terms.
        """
        if len(term_trackers) < 2:
            raise ValueError(f"A hybrid source needs at least two term trackers; got {len(term_trackers)}.")
        self._objective = objective  # READ-ONLY
        self._term_trackers = tuple(term_trackers)  # READ-ONLY

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
