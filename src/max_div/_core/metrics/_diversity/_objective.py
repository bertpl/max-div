"""Diversity objectives: what the solver maximizes, and the tie-breakers it ranks selections by.

A `DiversityObjective` is one of three shapes, each holding only the fields its value needs:

- `DiversityObjectiveSimple` — one diversity metric over one distance.
- `DiversityObjectiveHybridGeoMean` — the geometric mean of several simpler objectives (its terms).
- `DiversityObjectiveHybridFlattened` — one diversity metric over several distances at once, used
  for a tie-breaker.

The solver, its config, builders, presets and strategies read this type, never the bare
`DiversityMetric` enum, because an objective of several terms cannot be a single enum member.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from max_div._core.metrics._distance import DistanceMetric

from ._enum import DiversityContributionFamily, DiversityMetric

# A per-item diversity signal an objective reads: a distance paired with the contribution family a
# metric over it consumes.  The distance is `None` when it is the problem's own (given) distance.
DistanceFamilyPair = tuple[DistanceMetric | None, DiversityContributionFamily]


# =================================================================================================
#  DiversityObjective
# =================================================================================================
class DiversityObjective(ABC):
    """A diversity objective the solver maximizes, or a tie-breaker it ranks ties by.

    A subclass holds the fields its own value needs; the base gives the facts the solver reads off
    any objective — the distances and families it reads, and how to build a tie-breaker from it.
    """

    @abstractmethod
    def distance_family_pairs(self) -> tuple[DistanceFamilyPair, ...]:
        """Return the distinct (distance, contribution family) pairs this objective reads, in first-seen order.

        One pair is one tracker the solver builds; a pair repeated across terms is read once.
        """

    def distinct_distance_metrics(self) -> tuple[DistanceMetric | None, ...]:
        """Return the distinct distances this objective reads, in first-seen order."""
        return tuple(dict.fromkeys(distance_metric for distance_metric, _ in self.distance_family_pairs()))

    def has_single_separation_tracker(self) -> bool:
        """Return whether this objective reads exactly one tracker, of the separation family.

        The batched farthest-point construction is tailored to this case.
        """
        pairs = self.distance_family_pairs()
        return len(pairs) == 1 and pairs[0][1] == DiversityContributionFamily.SEPARATION

    def build_tie_breaker(self, tie_breaker_metric: DiversityMetric) -> DiversityObjectiveHybridFlattened:
        """Return the flattened tie-breaker of `tie_breaker_metric` over the distances this objective reads."""
        return DiversityObjectiveHybridFlattened(tie_breaker_metric, self.distinct_distance_metrics())


# =================================================================================================
#  Concrete objectives
# =================================================================================================
@dataclass(frozen=True)
class DiversityObjectiveSimple(DiversityObjective):
    """One diversity metric over one distance; `distance_metric` is `None` for the problem's own distance."""

    diversity_metric: DiversityMetric
    distance_metric: DistanceMetric | None = None

    def distance_family_pairs(self) -> tuple[DistanceFamilyPair, ...]:
        """Return the one (distance, family) pair this objective reads."""
        return ((self.distance_metric, self.diversity_metric.contribution_family),)


@dataclass(frozen=True)
class DiversityObjectiveHybridGeoMean(DiversityObjective):
    """The geometric mean of several diversity terms, each a simpler objective over its own distance."""

    terms: tuple[DiversityObjectiveSimple, ...]

    def __post_init__(self) -> None:
        """Reject fewer than two terms; a one-term geometric mean is a `DiversityObjectiveSimple`."""
        if len(self.terms) < 2:
            raise ValueError(f"A geometric-mean hybrid needs at least two terms; got {len(self.terms)}.")

    def distance_family_pairs(self) -> tuple[DistanceFamilyPair, ...]:
        """Return the distinct pairs of the terms, concatenated in first-seen order."""
        return tuple(dict.fromkeys(pair for term in self.terms for pair in term.distance_family_pairs()))


@dataclass(frozen=True)
class DiversityObjectiveHybridFlattened(DiversityObjective):
    """One diversity metric over several distances at once: the metric reads the distances' contributions joined.

    Holds one metric by construction, so a tie-breaker cannot mix metrics; the distances are
    `None` for the problem's own distance.
    """

    diversity_metric: DiversityMetric
    distance_metrics: tuple[DistanceMetric | None, ...]

    def distance_family_pairs(self) -> tuple[DistanceFamilyPair, ...]:
        """Return the distinct (distance, family) pairs, one per distinct distance, all of the metric's family."""
        family = self.diversity_metric.contribution_family
        return tuple(dict.fromkeys((distance_metric, family) for distance_metric in self.distance_metrics))


# =================================================================================================
#  Helpers
# =================================================================================================
def scoring_metric(objective: DiversityObjective) -> DiversityMetric:
    """Return the single diversity metric a score is computed with for `objective`.

    Defined for the single-metric objectives (`DiversityObjectiveSimple` and
    `DiversityObjectiveHybridFlattened`); a geometric-mean hybrid combines several metrics and is
    scored by its own reduction, not by a single metric.

    Raises:
        ValueError: If `objective` is a geometric-mean hybrid.
    """
    if isinstance(objective, (DiversityObjectiveSimple, DiversityObjectiveHybridFlattened)):
        return objective.diversity_metric
    raise ValueError(f"A geometric-mean hybrid has no single scoring metric: {objective}.")


def default_tie_breaker_metrics(diversity_metric: DiversityMetric) -> list[DiversityMetric]:
    """Return the tie-breaker metrics to score with when the caller sets none, by the main diversity metric.

    A near-degenerate main metric, where many selections share a score, gets tie-breakers that
    separate them; every other metric gets none.
    """
    if diversity_metric == DiversityMetric.MIN_SEPARATION:
        return [DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC]
    if diversity_metric in (DiversityMetric.GEOMEAN_SEPARATION, DiversityMetric.APPROX_GEOMEAN_SEPARATION):
        return [DiversityMetric.NON_ZERO_SEPARATION_FRAC]
    return []
