from __future__ import annotations

from typing import TYPE_CHECKING

from ._factory import build_diversity_contribution_tracker

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.metrics import DistanceAndFamily, DiversityObjective
    from max_div._core.metrics._distance import DistanceStore

    from ._base import DiversityContributionTracker

# `selected_contributions` reads this many slots without a loop, so a tracker set never holds more.
MAX_CONTRIBUTION_SLOTS = 2


# =================================================================================================
#  Contribution slots
# =================================================================================================
def build_diversity_contribution_slots(
    diversity_objective: DiversityObjective, diversity_tie_breakers: list[DiversityObjective]
) -> dict[DistanceAndFamily, int]:
    """Return each (distance, family) pair's slot: its index in the tuple that `selected_contributions` returns.

    A pair (`DistanceAndFamily`) joins a contribution family with the distance a term reads. The
    pairs are numbered in the order they first appear, across `diversity_objective` (the main
    objective, the one the solver maximizes) and then the tie-breakers, each pair once. Both the
    tracker set and the score generator take their slot assignments from this one function, so the
    tracker set returns its arrays in the same slot order that the score generator reads.
    """
    pairs = dict.fromkeys(
        pair
        for objective in (diversity_objective, *diversity_tie_breakers)
        for pair in objective.distance_and_family_pairs()
    )
    return {pair: slot for slot, pair in enumerate(pairs)}


# =================================================================================================
#  DiversityContributionTrackers
# =================================================================================================
class DiversityContributionTrackers:
    """The set of diversity-contribution trackers backing a solver state.

    Holds one tracker per (distance, family) pair that the objectives read, knows which pair is the
    main objective's, and applies every selection mutation to all trackers.
    """

    # -------------------------------------------------------------------------
    #  Construction & copy
    # -------------------------------------------------------------------------
    def __init__(
        self,
        trackers_by_pair: dict[DistanceAndFamily, DiversityContributionTracker],
        main_pair: DistanceAndFamily,
    ) -> None:
        """Initialize from an explicit pair -> tracker mapping; prefer the for_objectives() factory.

        Args:
            trackers_by_pair: (dict) one tracker per (distance, family) pair, in slot order (see
                `build_diversity_contribution_slots`); at most `MAX_CONTRIBUTION_SLOTS` of them.
            main_pair: (DistanceAndFamily) the (distance, family) pair of the main objective's tracker.

        Raises:
            ValueError: If the mapping holds more than `MAX_CONTRIBUTION_SLOTS` pairs.
        """
        if len(trackers_by_pair) > MAX_CONTRIBUTION_SLOTS:
            raise ValueError(
                f"A tracker set holds at most {MAX_CONTRIBUTION_SLOTS} (distance, family) pairs; "
                f"got {len(trackers_by_pair)}."
            )
        self._trackers_by_pair = trackers_by_pair  # READ-ONLY
        self._main_pair = main_pair  # READ-ONLY
        self._trackers = tuple(trackers_by_pair.values())  # in slot order
        self._main = trackers_by_pair[main_pair]

    @classmethod
    def for_objectives(
        cls,
        diversity_objective: DiversityObjective,
        diversity_tie_breakers: list[DiversityObjective],
        store: DistanceStore,
    ) -> DiversityContributionTrackers:
        """Build the tracker set that the objectives need, all reading `store`.

        The main objective's pair is slot 0 (see `build_diversity_contribution_slots`), and its
        tracker is `main`. Every tracker reads the one `store`; a pair's tracker is built for the
        pair's family.
        """
        slot_by_pair = build_diversity_contribution_slots(diversity_objective, diversity_tie_breakers)
        return cls(
            trackers_by_pair={
                (distance, family): build_diversity_contribution_tracker(family, store)
                for distance, family in slot_by_pair
            },
            main_pair=diversity_objective.distance_and_family_pairs()[0],
        )

    def copy(self) -> DiversityContributionTrackers:
        """Return a deep copy of this tracker set."""
        return DiversityContributionTrackers(
            trackers_by_pair={pair: tracker.copy() for pair, tracker in self._trackers_by_pair.items()},
            main_pair=self._main_pair,
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
    ) -> tuple[NDArray[np.float32], ...]:
        """Return the selected items' contribution values as a tuple, one array per slot, in slot order.

        The tuple is assembled without a loop, unrolled for the at most `MAX_CONTRIBUTION_SLOTS`
        trackers that `__init__` accepts, because this runs on the solver's innermost loop, once for
        every scored selection.

        The selection is passed twice on purpose: the trackers compute contributions from the mask,
        and the values are picked out by the index list, which costs O(n_selected) where picking by
        mask costs O(n).

        Args:
            selected: (n-sized bool ndarray) current selection mask.
            n_selected: (np.int32) number of True values in `selected`.
            selected_indices: (n_selected-sized int32 ndarray) the indices where `selected` is True.
        """
        trackers = self._trackers
        first = trackers[0].contribution_wrt_selection(selected, n_selected)[selected_indices]
        if len(trackers) == 1:
            return (first,)
        else:
            return first, trackers[1].contribution_wrt_selection(selected, n_selected)[selected_indices]
