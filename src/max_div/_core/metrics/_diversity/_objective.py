"""A diversity objective is what the solver maximizes; a tie-breaker is a further objective the solver ranks ties by.

A `DiversityObjective` is one of three kinds, each holding only the fields its value needs:

- `DiversityObjectiveSimple` — one diversity metric over one distance metric.
- `DiversityObjectiveHybridGeoMean` — the geometric mean of several simpler objectives (its terms).
- `DiversityObjectiveHybridFlattened` — one diversity metric over several distance metrics at once,
  read as one joined input; this is the shape of a tie-breaker.

Every kind computes its own diversity score (`compute`) from the per-item contributions the solver
tracks. The solver, its config, builders, presets and strategies read this type, never a bare
`DiversityMetric`, because an objective of several terms is not a single diversity metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ._enum import DiversityContributionFamily, DiversityMetric

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceMetric

    # the selected items' per-item contribution values, one array per tracked spec
    Contributions = Mapping["DiversityTrackerSpec", "NDArray[np.float32]"]


class DiversityTrackerSpec(NamedTuple):
    """The distance metric and the contribution family to which the diversity metric belongs.

    The solver builds one contribution tracker for each distinct spec.
    """

    distance_metric: DistanceMetric | None  # None → the problem's own distance
    contribution_family: DiversityContributionFamily


# =================================================================================================
#  DiversityObjective
# =================================================================================================
class DiversityObjective(ABC):
    """A diversity objective the solver maximizes, or a tie-breaker it ranks ties by.

    Each subclass holds the fields its kind needs and computes its own diversity score. The base
    derives, off the specs a subclass declares, the facts consumers read: the distinct distance
    metrics, and whether one separation tracker serves the objective.
    """

    @abstractmethod
    def compute(self, contributions: Contributions) -> float:
        """Return this objective's diversity score for the current selection.

        Args:
            contributions: the selected items' per-item contribution values, one array per tracked
                spec (see `tracker_specs`). Each of this objective's specs is a key.
        """

    @abstractmethod
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the distinct specs this objective reads, in first-seen order (one tracker each)."""

    @abstractmethod
    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the tie-breaker objectives to rank ties by when the caller sets none of its own."""

    def distinct_distance_metrics(self) -> tuple[DistanceMetric | None, ...]:
        """Return the distinct distance metrics this objective reads, in first-seen order."""
        return tuple(dict.fromkeys(spec.distance_metric for spec in self.tracker_specs()))

    def reads_single_separation_tracker(self) -> bool:
        """Return whether one separation tracker serves this objective: exactly one spec, of the separation family.

        The batched farthest-point construction and the SMART preset gate their fast path on this,
        since that path is written for a single separation tracker over the problem's own distance.
        """
        specs = self.tracker_specs()
        return len(specs) == 1 and specs[0].contribution_family == DiversityContributionFamily.SEPARATION


# =================================================================================================
#  Concrete objectives
# =================================================================================================
@dataclass(frozen=True)
class DiversityObjectiveSimple(DiversityObjective):
    """One diversity metric over one distance metric; `distance_metric` is `None` for the problem's own distance."""

    diversity_metric: DiversityMetric
    distance_metric: DistanceMetric | None = None

    def compute(self, contributions: Contributions) -> float:
        """Reduce this objective's one contribution array with its diversity metric."""
        return float(self.diversity_metric.compute(contributions[self._spec()]))

    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the one spec this objective reads."""
        return (self._spec(),)

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the separating tie-breakers a near-degenerate diversity metric needs; other metrics get none.

        A near-degenerate diversity metric, where many selections share a score, gets tie-breakers
        that separate them, each over this objective's own distance metric.
        """
        return [
            DiversityObjectiveHybridFlattened(tie_breaker_metric, (self.distance_metric,))
            for tie_breaker_metric in _separating_tie_breaker_metrics(self.diversity_metric)
        ]

    def _spec(self) -> DiversityTrackerSpec:
        """Return this objective's (distance metric, contribution family) spec."""
        return DiversityTrackerSpec(self.distance_metric, self.diversity_metric.contribution_family)


@dataclass(frozen=True)
class DiversityObjectiveHybridGeoMean(DiversityObjective):
    """The geometric mean of several diversity terms, each a simpler objective over its own distance metric."""

    terms: tuple[DiversityObjectiveSimple, ...]

    def __post_init__(self) -> None:
        """Reject fewer than two terms; a one-term geometric mean is a `DiversityObjectiveSimple`."""
        if len(self.terms) < 2:
            raise ValueError(f"A geometric-mean hybrid needs at least two terms; got {len(self.terms)}.")

    def compute(self, contributions: Contributions) -> float:
        """Return the geometric mean of the terms' diversity scores."""
        term_scores = np.array([term.compute(contributions) for term in self.terms], dtype=np.float64)
        return float(np.prod(term_scores) ** (1.0 / len(term_scores)))

    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the distinct specs of the terms, in first-seen order."""
        return tuple(dict.fromkeys(spec for term in self.terms for spec in term.tracker_specs()))

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the separating tie-breakers over the distinct distance metrics the terms read."""
        distances = self.distinct_distance_metrics()
        tie_breaker_metrics = (DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC)
        return [DiversityObjectiveHybridFlattened(metric, distances) for metric in tie_breaker_metrics]


@dataclass(frozen=True)
class DiversityObjectiveHybridFlattened(DiversityObjective):
    """One diversity metric over several distance metrics at once: it reads their contributions as one joined input.

    Holds one diversity metric by construction, so a tie-breaker cannot mix diversity metrics; a
    distance metric is `None` for the problem's own distance.
    """

    diversity_metric: DiversityMetric
    distance_metrics: tuple[DistanceMetric | None, ...]

    def compute(self, contributions: Contributions) -> float:
        """Reduce the joined contribution arrays of all this objective's specs with its diversity metric."""
        joined = np.concatenate([contributions[spec] for spec in self.tracker_specs()])
        return float(self.diversity_metric.compute(joined))

    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return one spec per distinct distance metric, all of the diversity metric's family."""
        family = self.diversity_metric.contribution_family
        return tuple(
            dict.fromkeys(DiversityTrackerSpec(distance_metric, family) for distance_metric in self.distance_metrics)
        )

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return none: a tie-breaker is not itself ranked by further tie-breakers."""
        return []


# =================================================================================================
#  Helpers
# =================================================================================================
def _separating_tie_breaker_metrics(diversity_metric: DiversityMetric) -> tuple[DiversityMetric, ...]:
    """Return the diversity metrics whose tie-breakers separate selections that a near-degenerate metric ties.

    A minimum-separation score depends only on the closest pair, so many selections share it; a
    geometric mean is zero as soon as one pair coincides, so many selections sit on that zero
    plateau. Each gets tie-breakers that tell such selections apart; every other metric gets none.
    """
    if diversity_metric == DiversityMetric.MIN_SEPARATION:
        return (DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC)
    if diversity_metric in (DiversityMetric.GEOMEAN_SEPARATION, DiversityMetric.APPROX_GEOMEAN_SEPARATION):
        return (DiversityMetric.NON_ZERO_SEPARATION_FRAC,)
    return ()
