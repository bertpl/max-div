"""`DiversityObjectiveBindings` binds one solve's diversity objectives to the stores and trackers they read."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from max_div._core.metrics import DistanceMetric, DiversityObjective, DiversityTrackerSpec


# =================================================================================================
#  DiversityObjectiveBindings
# =================================================================================================
@dataclass(frozen=True)
class DiversityObjectiveBindings:
    """The bindings map one solve's diversity objectives to one store per distance metric and one tracker per spec.

    The score reads one contribution array per tracker, in the trackers' order; the bindings also
    record where each objective's own arrays sit in that order. They are a function of the objective
    list alone, so every place that needs an order calls `for_objectives` and gets the same one:

    - the builder, for the stores it builds;
    - a worker, for the stores it attaches;
    - the solver state, for the trackers it builds and the arrays it scores.

    No layer derives an order of its own.
    """

    # one entry per distance store: the distinct distance metrics that the objectives use, in
    # first-seen order (`None` is the problem's own distance metric)
    distance_metrics: tuple[DistanceMetric | None, ...]
    # one entry per contribution tracker: the distinct specs (distance metric, contribution family)
    # that the objectives' contributions are tracked under, in first-seen order; this is also the
    # order of the score's contribution arrays
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
        distance_metrics = tuple(dict.fromkeys(spec.distance_metric for spec in tracker_specs))
        position_of_spec = {spec: position for position, spec in enumerate(tracker_specs)}
        objective_spec_positions = tuple(
            tuple(position_of_spec[spec] for spec in objective.tracker_specs) for objective in diversity_objectives
        )
        return cls(distance_metrics, tracker_specs, objective_spec_positions)
