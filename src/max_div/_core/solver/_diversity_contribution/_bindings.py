"""`DiversityObjectiveBindings` binds one solve's diversity objectives to the stores and trackers they read."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from max_div._core.metrics import DiversityObjective, DiversityTrackerSpec
    from max_div._core.metrics._distance import DistanceSpec


# ==================================================================================================
#  DiversityObjectiveBindings
# ==================================================================================================
@dataclass(frozen=True)
class DiversityObjectiveBindings:
    """The bindings map a solve's diversity objectives to a store per distance spec and a tracker per tracker spec.

    The score reads one contribution array per tracker, in the trackers' order; the bindings also
    record where each objective's own arrays sit in that order.

    The bindings are a function of the objective list alone, so each of these follows the one order
    that `for_objectives` returns:

    - the distance stores;
    - the contribution trackers;
    - the score's contribution arrays.

    No layer derives an order of its own.
    """

    # `distance_specs` has one entry per distance store: the distinct distance specs that the objectives
    # use, in first-seen order.
    distance_specs: tuple[DistanceSpec, ...]
    # `tracker_specs` has one entry per contribution tracker: the distinct tracker specs (distance spec,
    # contribution family) that the objectives' contributions are tracked under, in first-seen order;
    # this is also the order of the score's contribution arrays.
    tracker_specs: tuple[DiversityTrackerSpec, ...]
    # one entry per objective, in objective order: the positions in `tracker_specs` of the specs that
    # the objective's score reads, in that objective's own spec order
    objective_spec_positions: tuple[tuple[int, ...], ...]

    @classmethod
    def for_objectives(cls, diversity_objectives: Sequence[DiversityObjective]) -> DiversityObjectiveBindings:
        """Return the bindings of the objectives, listed with the primary objective first."""
        tracker_specs = tuple(
            dict.fromkeys(spec for objective in diversity_objectives for spec in objective.tracker_specs)
        )
        distance_specs = tuple(dict.fromkeys(spec.distance_spec for spec in tracker_specs))
        position_of_spec = {spec: position for position, spec in enumerate(tracker_specs)}
        objective_spec_positions = tuple(
            tuple(position_of_spec[spec] for spec in objective.tracker_specs) for objective in diversity_objectives
        )
        return cls(distance_specs, tracker_specs, objective_spec_positions)
