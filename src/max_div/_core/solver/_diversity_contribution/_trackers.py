from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from max_div._core.metrics import DiversityContributionFamily

from ._factory import build_diversity_contribution_tracker

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from max_div._core.metrics import DiversityMetric
    from max_div._core.metrics._distance import DistanceStore

    from ._base import DiversityContributionTracker


# =================================================================================================
#  SelectedContributions
# =================================================================================================
# A selection's per-family contribution values of the *selected* items: a fixed-order tuple with
# one slot per DiversityContributionFamily, in enum definition order (see selected_contributions_slot).
# Families no metric consumes hold a shared empty array.  A plain tuple instead of a value object
# keeps this hot-path payload as cheap as possible.
SelectedContributions = tuple["NDArray[np.float32]", "NDArray[np.float32]"]

_FAMILY_SLOTS = {family: slot for slot, family in enumerate(DiversityContributionFamily)}


def selected_contributions_slot(family: DiversityContributionFamily) -> int:
    """Return the SelectedContributions tuple slot holding the given family's contribution values."""
    return _FAMILY_SLOTS[family]


# =================================================================================================
#  DiversityContributionTrackers
# =================================================================================================
class DiversityContributionTrackers:
    """The set of diversity-contribution trackers backing a solver state.

    Holds one tracker per contribution family needed by the configured metrics (families no metric
    consumes are simply absent, so they cost nothing to maintain), knows which family is the main
    diversity metric's, and fans every selection mutation out to all trackers.
    """

    # -------------------------------------------------------------------------
    #  Construction & copy
    # -------------------------------------------------------------------------
    def __init__(
        self,
        trackers_by_family: dict[DiversityContributionFamily, DiversityContributionTracker],
        main_family: DiversityContributionFamily,
    ) -> None:
        """Initialize from an explicit family -> tracker mapping; prefer the for_metrics() factory.

        Args:
            trackers_by_family: (dict) one tracker per tracked contribution family.
            main_family: (DiversityContributionFamily) family of the main diversity metric.
        """
        self._trackers_by_family = trackers_by_family  # READ-ONLY
        self._main_family = main_family  # READ-ONLY
        self._trackers = tuple(trackers_by_family.values())  # iteration order for mutation fan-out
        self._main = trackers_by_family[main_family]
        # per-family trackers for scoring reads (None = family not tracked)
        self._separation_tracker = trackers_by_family.get(DiversityContributionFamily.SEPARATION)
        self._mean_distance_tracker = trackers_by_family.get(DiversityContributionFamily.MEAN_DISTANCE)

    @classmethod
    def for_metrics(
        cls,
        diversity_metric: DiversityMetric,
        diversity_tie_breakers: list[DiversityMetric],
        store: DistanceStore,
    ) -> DiversityContributionTrackers:
        """Build the tracker set required by the given metrics (main metric's family first).

        Args:
            diversity_metric: (DiversityMetric) the main diversity metric.
            diversity_tie_breakers: (list[DiversityMetric]) the configured tie-breaker metrics.
            store: (DistanceStore) pairwise-distance storage the trackers read from.
        """
        main_family = diversity_metric.contribution_family
        families = dict.fromkeys([main_family, *(tb.contribution_family for tb in diversity_tie_breakers)])
        return cls(
            trackers_by_family={family: build_diversity_contribution_tracker(family, store) for family in families},
            main_family=main_family,
        )

    def copy(self) -> DiversityContributionTrackers:
        """Return a deep copy of this tracker set."""
        return DiversityContributionTrackers(
            trackers_by_family={family: t.copy() for family, t in self._trackers_by_family.items()},
            main_family=self._main_family,
        )

    # -------------------------------------------------------------------------
    #  Main tracker
    # -------------------------------------------------------------------------
    @property
    def main(self) -> DiversityContributionTracker:
        """Return the main diversity metric's tracker (feeds the strategy-facing contribution reads)."""
        return self._main

    # -------------------------------------------------------------------------
    #  Mutation fan-out
    # -------------------------------------------------------------------------
    def add(self, index: np.int32) -> None:
        """Update all trackers after adding point `index` to the selection."""
        for tracker in self._trackers:
            tracker.add(index)

    def add_many(self, indices: NDArray[np.int32], parallel: bool = False) -> None:
        """Update all trackers after adding all points in `indices`; see the base class on `parallel`."""
        for tracker in self._trackers:
            tracker.add_many(indices, parallel=parallel)

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Update all trackers after removing point `index` from the selection."""
        for tracker in self._trackers:
            tracker.remove(index, new_selection)

    def remove_many(self, indices: NDArray[np.int32], new_selection: NDArray[np.int32]) -> None:
        """Update all trackers after removing all points in `indices` from the selection."""
        for tracker in self._trackers:
            tracker.remove_many(indices, new_selection)

    def reset(self) -> None:
        """Reset all trackers to the empty selection; see the base class for the snapshot caveat."""
        for tracker in self._trackers:
            tracker.reset()

    # -------------------------------------------------------------------------
    #  Snapshot
    # -------------------------------------------------------------------------
    def push_snapshot(self) -> None:
        """Save the current contribution state of all trackers on top of their snapshot stacks."""
        for tracker in self._trackers:
            tracker.push_snapshot()

    def pop_snapshot(self, restore: bool) -> None:
        """Discard every tracker's top snapshot, first restoring from it if `restore`."""
        for tracker in self._trackers:
            tracker.pop_snapshot(restore)

    # -------------------------------------------------------------------------
    #  Scoring reads
    # -------------------------------------------------------------------------
    def selected_contributions(
        self, selected: NDArray[np.bool], n_selected: np.int32, selected_indices: NDArray[np.int32]
    ) -> SelectedContributions:
        """Return the selected items' contribution values, one SelectedContributions slot per family.

        Slots of families this set does not track hold a shared empty array (never read, since the
        score generator only consumes the families its metrics were bound to).

        The selection is passed twice on purpose: the trackers compute contributions from the mask,
        and the values are picked out by the index list, which costs O(n_selected) where picking by
        mask costs O(n).

        Args:
            selected: (n-sized bool ndarray) current selection mask.
            n_selected: (np.int32) number of True values in `selected`.
            selected_indices: (n_selected-sized int32 ndarray) the indices where `selected` is True.
        """
        sep_tracker = self._separation_tracker
        mean_tracker = self._mean_distance_tracker
        if sep_tracker is None:
            sep = _EMPTY_NP_ARRAY_FLOAT32
        else:
            sep = sep_tracker.contribution_wrt_selection(selected, n_selected)[selected_indices]
        if mean_tracker is None:
            mean = _EMPTY_NP_ARRAY_FLOAT32
        else:
            mean = mean_tracker.contribution_wrt_selection(selected, n_selected)[selected_indices]
        return sep, mean


# shared placeholder for the contribution values of untracked families
_EMPTY_NP_ARRAY_FLOAT32 = np.array([], dtype=np.float32)
