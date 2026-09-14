"""The bindings tie one solve's diversity objectives to what they read; see `DiversityObjectiveBindings`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from max_div._core.metrics import DiversityObjective, DiversityTrackerSpec
    from max_div._core.solver._distance_storage import StoreDistance


# =================================================================================================
#  DiversityObjectiveBindings
# =================================================================================================
@dataclass(frozen=True)
class DiversityObjectiveBindings:
    """What one solve's diversity objectives are bound to: a store per distinct distance, a tracker per distinct spec.

    The score reads one contribution array per tracker, in the trackers' order; the bindings also
    record where each objective's own arrays sit in that order. They are a function of the objective
    list alone, so every place that needs an order calls `for_objectives` and gets the same one:

    - the builder, for the stores it builds;
    - a worker, for the stores it attaches;
    - the solver state, for the trackers it builds and the arrays it scores.

    No layer derives an order of its own.
    """

    # the distinct distances the objectives read, in first-seen order: one store each
    # (`None` is the problem's own distance)
    store_distances: tuple[StoreDistance, ...]
    # the distinct specs the objectives read, in first-seen order: one tracker each, and the order
    # in which the score's contribution arrays are passed
    tracker_specs: tuple[DiversityTrackerSpec, ...]
    # per objective, in objective order: the positions in `tracker_specs` of the specs it reads, in
    # its own spec order
    objective_spec_positions: tuple[tuple[int, ...], ...]

    @classmethod
    def for_objectives(cls, diversity_objectives: Sequence[DiversityObjective]) -> DiversityObjectiveBindings:
        """Return the bindings of the objectives, listed with the primary objective first."""
        tracker_specs = tuple(
            dict.fromkeys(spec for objective in diversity_objectives for spec in objective.tracker_specs)
        )
        store_distances = tuple(dict.fromkeys(spec.distance_metric for spec in tracker_specs))
        position_of_spec = {spec: position for position, spec in enumerate(tracker_specs)}
        objective_spec_positions = tuple(
            tuple(position_of_spec[spec] for spec in objective.tracker_specs) for objective in diversity_objectives
        )
        return cls(store_distances, tracker_specs, objective_spec_positions)
