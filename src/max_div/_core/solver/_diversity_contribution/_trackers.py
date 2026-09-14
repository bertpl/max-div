from __future__ import annotations

from typing import TYPE_CHECKING

from max_div._core.metrics import DiversityObjectiveHybrid

from ._factory import build_diversity_contribution_tracker
from ._hybrid_source import HybridPerItemContributionSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.metrics import DistanceMetric, DiversityObjective, DiversityTrackerSpec
    from max_div._core.metrics._distance import DistanceStore

    from ._base import DiversityContributionTracker, PerItemContributionSource


# =================================================================================================
#  DiversityContributionTrackers
# =================================================================================================
class DiversityContributionTrackers:
    """The set of diversity-contribution trackers backing a solver state.

    Holds one tracker per spec, applies every selection mutation to all trackers, and, given the
    objective the solver maximizes, returns the source of the per-item contributions the strategies
    read: that objective's one tracker, or a hybrid source over its term trackers.
    """

    # -------------------------------------------------------------------------
    #  Construction
    # -------------------------------------------------------------------------
    def __init__(self, trackers_by_spec: dict[DiversityTrackerSpec, DiversityContributionTracker]) -> None:
        """Initialize from an explicit spec -> tracker mapping; prefer the for_specs() factory.

        Args:
            trackers_by_spec: (dict) one tracker per spec the objectives read, in the bindings' spec
                order (`DiversityObjectiveBindings`). `per_item_contribution_source_for` takes positions
                that index into this order.
        """
        self._trackers_by_spec = trackers_by_spec  # READ-ONLY
        self._trackers = tuple(trackers_by_spec.values())  # iteration order for mutation fan-out

    @classmethod
    def for_specs(
        cls,
        tracker_specs: Sequence[DiversityTrackerSpec],
        stores_by_distance: Mapping[DistanceMetric | None, DistanceStore],
    ) -> DiversityContributionTrackers:
        """Build one tracker per spec, in the given order, each over the store of its distance.

        `tracker_specs` is the bindings' spec order (see `DiversityObjectiveBindings`).
        `stores_by_distance` maps a spec's distance metric (`None` for the problem's own) to the store
        the spec's tracker reads.
        """
        return cls(
            trackers_by_spec={
                spec: build_diversity_contribution_tracker(
                    spec.contribution_family, stores_by_distance[spec.distance_metric]
                )
                for spec in tracker_specs
            }
        )

    # -------------------------------------------------------------------------
    #  Per-item contribution source
    # -------------------------------------------------------------------------
    def per_item_contribution_source_for(
        self, objective: DiversityObjective, spec_positions: Sequence[int]
    ) -> PerItemContributionSource:
        """Return the source of `objective`'s per-item contribution: its one tracker, or a hybrid source.

        Args:
            objective: the objective the solver maximizes.
            spec_positions: for each of `objective`'s specs (in that objective's order, repeats kept), the
                position of that spec's tracker in this set: that objective's entry of the bindings'
                `objective_spec_positions`.

        A simple objective gets its one tracker as the source itself: its per-item contribution is that
        tracker's contribution array, and wrapping it in a source would only add a function call
        whenever the contribution is read.
        """
        term_trackers = [self._trackers[position] for position in spec_positions]
        if isinstance(objective, DiversityObjectiveHybrid):
            return HybridPerItemContributionSource(objective, term_trackers)
        else:
            return term_trackers[0]

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
    @property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the specs of the trackers in this set, in the order `selected_contributions` returns their arrays."""
        return tuple(self._trackers_by_spec)

    def selected_contributions(
        self, selected: NDArray[np.bool], n_selected: np.int32, selected_indices: NDArray[np.int32]
    ) -> list[NDArray[np.float32]]:
        """Return the selected items' contribution values, one array per tracker, in the order of `tracker_specs`.

        The selection is passed twice on purpose: the trackers compute contributions from the mask,
        and the values are picked out by the index list, which costs O(n_selected) where picking by
        mask costs O(n).

        Args:
            selected: (n-sized bool ndarray) current selection mask.
            n_selected: (np.int32) number of True values in `selected`.
            selected_indices: (n_selected-sized int32 ndarray) the indices where `selected` is True.
        """
        return [
            tracker.contribution_wrt_selection(selected, n_selected)[selected_indices] for tracker in self._trackers
        ]
