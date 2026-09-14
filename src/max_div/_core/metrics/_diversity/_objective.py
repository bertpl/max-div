"""A diversity objective is what the solver maximizes; a tie-breaker is a further objective the solver ranks ties by.

A `DiversityObjective` is one of two kinds, each holding only the fields that kind of objective needs:

- `DiversityObjectiveSimple` — one diversity metric over one distance metric.
- `DiversityObjectiveHybrid` — several simple objectives (its terms) combined by a geometric or an
  arithmetic mean; a hybrid objective's tie-breakers are hybrids too.

Every kind computes its own diversity score (`compute`) from the per-item contributions the solver
tracks. The solver passes `compute` one array per spec of `tracker_specs`, in that order; a hybrid
lists one spec per term, so two terms over one spec receive the same array twice. The
solver, its config, builders, presets and strategies read this type, never a bare `DiversityMetric`,
because an objective of several terms is not a single diversity metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from max_div._core._math.geomean import geomean_f32

from ._enum import DiversityContributionFamily, DiversityMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceMetric


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

    Each subclass holds the fields its kind needs and computes its own diversity score. From the
    specs a subclass declares, the base derives the distinct specs and the distinct distance metrics.
    """

    @abstractmethod
    def compute(self, contributions: Sequence[NDArray[np.float32]]) -> float:
        """Return this objective's diversity score for the current selection.

        Runs once per scored selection, so it and its cached inputs must stay cheap.

        Args:
            contributions: the selected items' per-item contribution values, one array per spec of
                this objective, in the order of `tracker_specs`.
        """

    @property
    @abstractmethod
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return one spec per array `compute` takes, in that order; a spec may repeat."""

    @abstractmethod
    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the tie-breaker objectives to rank ties by when the caller sets none of its own."""

    @cached_property
    def distinct_tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the distinct specs of this objective, in first-seen order."""
        return tuple(dict.fromkeys(self.tracker_specs))

    def distinct_distance_metrics(self) -> tuple[DistanceMetric | None, ...]:
        """Return the distinct distance metrics of this objective's specs, in first-seen order."""
        return tuple(dict.fromkeys(spec.distance_metric for spec in self.tracker_specs))


# =================================================================================================
#  Concrete objectives
# =================================================================================================
@dataclass(frozen=True)
class DiversityObjectiveSimple(DiversityObjective):
    """One diversity metric over one distance metric; `distance_metric` is `None` for the problem's own distance."""

    diversity_metric: DiversityMetric
    distance_metric: DistanceMetric | None = None

    def compute(self, contributions: Sequence[NDArray[np.float32]]) -> float:
        """Reduce this objective's one contribution array with its diversity metric."""
        return float(self.diversity_metric.compute(contributions[0]))

    @cached_property
    def tracker_spec(self) -> DiversityTrackerSpec:
        """Return the one spec of this objective: the single entry of `tracker_specs`, as a shorthand."""
        return DiversityTrackerSpec(self.distance_metric, self.diversity_metric.contribution_family)

    @cached_property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the one spec of this objective."""
        return (self.tracker_spec,)

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the separating tie-breakers a near-degenerate diversity metric needs; other metrics get none.

        A near-degenerate diversity metric, where many selections share a score, gets tie-breakers
        that separate them, each over this objective's own distance metric.
        """
        return [
            DiversityObjectiveSimple(tie_breaker_metric, self.distance_metric)
            for tie_breaker_metric in _separating_tie_breaker_metrics(self.diversity_metric)
        ]


class HybridObjectiveType(StrEnum):
    """How a hybrid objective combines its terms: by their geometric mean or by their arithmetic mean."""

    GEOMETRIC_MEAN = "GEOMETRIC_MEAN"
    ARITHMETIC_MEAN = "ARITHMETIC_MEAN"


@dataclass(frozen=True)
class DiversityObjectiveHybrid(DiversityObjective):
    """Several simple objectives (its terms) combined by their geometric or arithmetic mean.

    Terms are simple objectives only, so each term reads exactly one of the arrays passed to
    `compute`. The solver maximizes a hybrid with the geometric combination; a hybrid's default
    tie-breakers are hybrids of one tie-breaker metric over the distinct distance metrics.
    """

    terms: tuple[DiversityObjectiveSimple, ...]
    combination: HybridObjectiveType = HybridObjectiveType.GEOMETRIC_MEAN

    def __post_init__(self) -> None:
        """Reject fewer than two terms and any term that is not a simple objective.

        A one-term hybrid is a `DiversityObjectiveSimple`.
        """
        if len(self.terms) < 2:
            raise ValueError(f"A hybrid objective needs at least two terms; got {len(self.terms)}.")
        for term in self.terms:
            if not isinstance(term, DiversityObjectiveSimple):
                raise TypeError(f"A hybrid objective's terms must be simple objectives; got {type(term).__name__}.")

    def compute(self, contributions: Sequence[NDArray[np.float32]]) -> float:
        """Return the combination of the terms' diversity scores, each term reading its own array."""
        term_scores = np.array(
            [term.compute((contribution,)) for term, contribution in zip(self.terms, contributions, strict=True)],
            dtype=np.float32,
        )
        if self.combination == HybridObjectiveType.GEOMETRIC_MEAN:
            return float(geomean_f32(term_scores))
        else:
            return float(np.mean(term_scores))

    @cached_property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the terms' specs in term order, a spec repeated once per term that has it."""
        return tuple(term.tracker_spec for term in self.terms)

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the separating tie-breakers over the terms' distinct distance metrics.

        A geometric mean has the plateaus that need breaking: it is pulled to zero by a single zero
        term, and above zero it can tie. An arithmetic mean has neither, since every term keeps
        contributing and one zero term only lowers it, so an arithmetic hybrid gets no tie-breakers.
        """
        if self.combination == HybridObjectiveType.ARITHMETIC_MEAN:
            return []
        else:
            distances = self.distinct_distance_metrics()
            return [
                # the approximate geometric mean falls the more separations are zero, so a selection with
                # fewer zero separations ranks higher; combined geometrically, one zero distance pulls it
                # down without pinning the tie-breaker at zero the way an exact geometric mean would
                DiversityObjectiveHybrid(
                    tuple(DiversityObjectiveSimple(DiversityMetric.APPROX_GEOMEAN_SEPARATION, d) for d in distances),
                    HybridObjectiveType.GEOMETRIC_MEAN,
                ),
                # once the approximate geometric mean has underflowed to zero, the non-zero fraction still
                # counts the coincident pairs; combined arithmetically, a distance with no non-zero
                # separation only lowers it, so fixing another distance still ranks higher
                DiversityObjectiveHybrid(
                    tuple(DiversityObjectiveSimple(DiversityMetric.NON_ZERO_SEPARATION_FRAC, d) for d in distances),
                    HybridObjectiveType.ARITHMETIC_MEAN,
                ),
            ]


# =================================================================================================
#  Helpers
# =================================================================================================
def _separating_tie_breaker_metrics(diversity_metric: DiversityMetric) -> tuple[DiversityMetric, ...]:
    """Return the tie-breaker diversity metrics for a primary metric that has defaults; empty for the rest."""
    if diversity_metric == DiversityMetric.MIN_SEPARATION:
        # min-separation reacts only to the closest pair; the approximate geomean rewards a uniform
        # spread, which opens room around that pair so the minimum separation itself can grow.
        return (DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC)
    if diversity_metric in (DiversityMetric.GEOMEAN_SEPARATION, DiversityMetric.APPROX_GEOMEAN_SEPARATION):
        # once more than one pair coincides the geomean is stuck at zero; the non-zero fraction
        # rewards cutting the count of coincident pairs, a path back toward a non-zero geomean.
        return (DiversityMetric.NON_ZERO_SEPARATION_FRAC,)
    return ()
