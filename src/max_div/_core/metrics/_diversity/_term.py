"""A diversity term pairs a diversity metric with the distance it is measured over."""

from dataclasses import dataclass

from max_div._core.metrics._distance import DistanceMetric

from ._enum import DiversityMetric


@dataclass(frozen=True)
class DiversityTerm:
    """One term of a diversity objective: a `DiversityMetric` measured over a distance.

    A `None` distance means the problem's given distance — the vector problem's `distance_metric`,
    or a precomputed-distances problem's matrix. Binding `None` to a concrete distance store is the
    distance-store layer's job, not this term's.
    """

    metric: DiversityMetric
    distance: DistanceMetric | None = None
