from __future__ import annotations

from typing import TYPE_CHECKING

from ._factory import build_diversity_contribution_tracker

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.metrics import DiversityObjective, DiversityTrackerSpec
    from max_div._core.metrics._distance import DistanceStore

    from ._base import DiversityContributionTracker


# =================================================================================================
#  DiversityContributionTrackers
# =================================================================================================
class DiversityContributionTrackers:
    """The set of diversity-contribution trackers backing a solver state.

    Holds one tracker per spec that the objectives read, knows which spec is the main objective's,
    and applies every selection mutation to all trackers.
    """

    # -------------------------------------------------------------------------
    #  Construction & copy
    # -------------------------------------------------------------------------
    def __init__(
        self,
        trackers_by_spec: dict[DiversityTrackerSpec, DiversityContributionTracker],
        main_spec: DiversityTrackerSpec,
    ) -> None:
        """Initialize from an explicit spec -> tracker mapping; prefer the for_objectives() factory.

        Args:
            trackers_by_spec: (dict) one tracker per spec the objectives read.
            main_spec: (DiversityTrackerSpec) the spec of the main objective's tracker.
        """
        self._trackers_by_spec = trackers_by_spec  # READ-ONLY
        self._main_spec = main_spec  # READ-ONLY
        self._trackers = tuple(trackers_by_spec.values())  # iteration order for mutation fan-out
        self._main = trackers_by_spec[main_spec]

    @classmethod
    def for_objectives(
        cls,
        diversity_objective: DiversityObjective,
        diversity_tie_breakers: list[DiversityObjective],
        store: DistanceStore,
    ) -> DiversityContributionTrackers:
        """Build the tracker set that the objectives need, all reading `store`.

        The set holds one tracker per distinct spec the objectives read — the main objective's specs
        first, then the tie-breakers' — and `main` is the tracker of the main objective's first spec.
        Every tracker reads the one `store`; a spec's tracker is built for the spec's contribution family.
        """
        specs = dict.fromkeys(
            spec for objective in (diversity_objective, *diversity_tie_breakers) for spec in objective.tracker_specs()
        )
        return cls(
            trackers_by_spec={
                spec: build_diversity_contribution_tracker(spec.contribution_family, store) for spec in specs
            },
            main_spec=diversity_objective.tracker_specs()[0],
        )

    def copy(self) -> DiversityContributionTrackers:
        """Return a deep copy of this tracker set."""
        return DiversityContributionTrackers(
            trackers_by_spec={spec: tracker.copy() for spec, tracker in self._trackers_by_spec.items()},
            main_spec=self._main_spec,
        )

    # -------------------------------------------------------------------------
    #  Main tracker
    # -------------------------------------------------------------------------
    @property
    def main(self) -> DiversityContributionTracker:
        """Return the main objective's tracker, whose contribution values the strategies read."""
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

    def remove_trial(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        """Apply the selected-only removal update on every tracker; see DiversityContributionTracker.remove_trial."""
        for tracker in self._trackers:
            tracker.remove_trial(index, new_selection)

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
    ) -> dict[DiversityTrackerSpec, NDArray[np.float32]]:
        """Return the selected items' contribution values, one array per tracked spec, keyed by spec.

        Each objective's `compute` reads the arrays of its own specs from this mapping.

        The selection is passed twice on purpose: the trackers compute contributions from the mask,
        and the values are picked out by the index list, which costs O(n_selected) where picking by
        mask costs O(n).

        Args:
            selected: (n-sized bool ndarray) current selection mask.
            n_selected: (np.int32) number of True values in `selected`.
            selected_indices: (n_selected-sized int32 ndarray) the indices where `selected` is True.
        """
        return {
            spec: tracker.contribution_wrt_selection(selected, n_selected)[selected_indices]
            for spec, tracker in self._trackers_by_spec.items()
        }
