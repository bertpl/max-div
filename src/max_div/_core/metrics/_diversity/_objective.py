"""A diversity objective is what the solver maximizes; a tie-breaker is a further objective the solver ranks ties by.

A `DiversityObjective` is one of two kinds, each holding only the fields that kind of objective needs:

- `DiversityObjectiveSimple` — one diversity metric over one set of distances, named by a `DistanceSpec`.
- `DiversityObjectiveHybrid` — several simple objectives (its terms) combined by a weighted aggregation,
  a `HybridAggregationBase`.

Every kind computes its own diversity score (`compute`) from the per-item contributions the solver
tracks. The solver passes `compute` one array per spec of `tracker_specs`, in that order; a hybrid
lists one spec per term, so two terms over one spec receive the same array twice. Every kind also
computes, from those same per-spec arrays, its own per-item contribution for all items
(`compute_per_item_contributions`), the value the solver samples items by. The solver, its config, builders, presets
and strategies read this type, never a bare `DiversityMetric`, because an objective of several
terms is not a single diversity metric.

The default tie-breakers follow one rule for both kinds, over the diversity metrics of the terms (a
simple objective counting as a single term):

- the approximate geomean is needed when a term is min-separation;
- the non-zero fraction is needed when a term is min-separation or goes to zero when one pair coincides.

Each tie-breaker is computed over every distinct distance spec of the objective, as a hybrid when there are several.
A hybrid gets one more tie-breaker, ranked before these, when its aggregation returns a tie-breaker
aggregation over the hybrid's own terms (`HybridAggregationBase.tie_breaker_aggregation_over_terms`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ._aggregation import HybridAggregationArithmeticMean, HybridAggregationBase, HybridAggregationGeometricMean
from ._enum import DiversityContributionFamily, DiversityMetric

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from max_div._core.metrics._distance import DistanceSpec


class DiversityTrackerSpec(NamedTuple):
    """A tracker spec pairs the distance spec that a diversity metric reads with the metric's contribution family.

    The solver builds one contribution tracker for each distinct tracker spec.
    """

    distance_spec: DistanceSpec
    contribution_family: DiversityContributionFamily


# ==================================================================================================
#  DiversityObjective
# ==================================================================================================
class DiversityObjective(ABC):
    """A diversity objective the solver maximizes, or a tie-breaker it ranks ties by.

    Each subclass holds the fields its kind needs and computes its own diversity score. From the
    tracker specs that a subclass declares, the base derives the distinct tracker specs and the distinct
    distance specs.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """Return a short label naming the metric and its distance, in the form of `HybridDiversityMetric.label`."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, contributions: Sequence[NDArray[np.float32]]) -> float:
        """Return this objective's diversity score for the current selection.

        Runs once per scored selection, so it and its cached inputs must stay cheap.

        Args:
            contributions: the selected items' per-item contribution values, one array per spec of
                this objective, in the order of `tracker_specs`.
        """

    @abstractmethod
    def compute_per_item_contributions(self, contributions: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Return this objective's per-item contribution of every item: the value the solver samples items by.

        Args:
            contributions: all items' per-item contribution values, one array per spec of this
                objective, in the order of `tracker_specs`.
        """

    @property
    @abstractmethod
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return one spec per array `compute` takes, in that order; a spec may repeat."""

    @property
    @abstractmethod
    def diversity_metrics(self) -> tuple[DiversityMetric, ...]:
        """Return the diversity metric of each term, in term order; a simple objective is a single term."""

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the tie-breaker objectives to rank ties by when the caller sets none of its own.

        The rule in the module docstring, applied to this objective's diversity metrics; each
        tie-breaker is built over this objective's distinct distance specs.
        """
        # --- which tie-breakers the metrics need ----
        # min-separation depends on the closest pair alone, so a swap that spreads the other items
        # leaves the score unchanged; such a swap has value, though: it frees room around the closest
        # pair and makes a later swap that moves one of its items apart more likely. The approximate
        # geomean rewards it.
        needs_approx_geomean = DiversityMetric.MIN_SEPARATION in self.diversity_metrics
        # the geometric and harmonic means are zero as soon as one pair coincides, so once two pairs
        # coincide no single swap moves the score off zero; the non-zero fraction counts the coincident
        # pairs down. A min-separation objective needs it too, for when the approximate geomean has
        # underflowed to zero.
        zero_pinned_metrics = {
            DiversityMetric.GEOMEAN_SEPARATION,
            DiversityMetric.APPROX_GEOMEAN_SEPARATION,
            DiversityMetric.HARMONIC_MEAN_SEPARATION,
        }
        needs_non_zero_frac = needs_approx_geomean or any(
            metric in zero_pinned_metrics for metric in self.diversity_metrics
        )

        # --- the tie-breaker metrics, in rank order --
        # a hybrid tie-breaker aggregates its per-distance terms geometrically for the approximate
        # geomean and arithmetically for the non-zero fraction: on the arithmetic one, a distance with
        # no non-zero separation lowers the tie-breaker without making it zero
        tie_breaker_metrics: list[tuple[DiversityMetric, type[HybridAggregationBase]]] = []
        if needs_approx_geomean:
            tie_breaker_metrics.append((DiversityMetric.APPROX_GEOMEAN_SEPARATION, HybridAggregationGeometricMean))
        if needs_non_zero_frac:
            tie_breaker_metrics.append((DiversityMetric.NON_ZERO_SEPARATION_FRAC, HybridAggregationArithmeticMean))

        # --- each over every distinct distance ------
        distance_specs = self.distinct_distance_specs()
        tie_breakers: list[DiversityObjective] = []
        for metric, aggregation_type in tie_breaker_metrics:
            if len(distance_specs) == 1:
                tie_breakers.append(DiversityObjectiveSimple(metric, distance_specs[0]))
            else:
                terms = tuple(DiversityObjectiveSimple(metric, distance_spec) for distance_spec in distance_specs)
                tie_breakers.append(DiversityObjectiveHybrid(terms, aggregation_type.with_unit_weights(len(terms))))
        return tie_breakers

    @abstractmethod
    def with_distance_specs(self, replacement_specs: Mapping[DistanceSpec, DistanceSpec]) -> DiversityObjective:
        """Return a copy of this objective with each distance spec replaced by its entry in `replacement_specs`.

        Args:
            replacement_specs: a replacement for every distance spec of this objective.
        """

    @cached_property
    def distinct_tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the distinct specs of this objective, in first-seen order."""
        return tuple(dict.fromkeys(self.tracker_specs))

    def distinct_distance_specs(self) -> tuple[DistanceSpec, ...]:
        """Return the distinct distance specs of this objective's tracker specs, in first-seen order."""
        return tuple(dict.fromkeys(spec.distance_spec for spec in self.tracker_specs))


# ==================================================================================================
#  Concrete objectives
# ==================================================================================================
@dataclass(frozen=True)
class DiversityObjectiveSimple(DiversityObjective):
    """A simple objective applies one diversity metric to the distances that one distance spec names."""

    diversity_metric: DiversityMetric
    distance_spec: DistanceSpec

    @property
    def label(self) -> str:
        """Return e.g. `MIN_SEPARATION over L2`, or `MIN_SEPARATION over user distances`."""
        return f"{self.diversity_metric.value} over {self.distance_spec.label}"

    def with_distance_specs(self, replacement_specs: Mapping[DistanceSpec, DistanceSpec]) -> DiversityObjectiveSimple:
        """Return a copy of this objective with its distance spec replaced by its entry in `replacement_specs`."""
        return replace(self, distance_spec=replacement_specs[self.distance_spec])

    def compute(self, contributions: Sequence[NDArray[np.float32]]) -> float:
        """Reduce this objective's one contribution array with its diversity metric."""
        return float(self.diversity_metric.compute(contributions[0]))

    def compute_per_item_contributions(self, contributions: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Return the one array unchanged: a single spec's contribution is the objective's own contribution."""
        return contributions[0]

    @cached_property
    def tracker_spec(self) -> DiversityTrackerSpec:
        """Return the one spec of this objective: the single entry of `tracker_specs`, as a shorthand."""
        return DiversityTrackerSpec(self.distance_spec, self.diversity_metric.contribution_family)

    @cached_property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the one spec of this objective."""
        return (self.tracker_spec,)

    @cached_property
    def diversity_metrics(self) -> tuple[DiversityMetric, ...]:
        """Return the one diversity metric of this objective."""
        return (self.diversity_metric,)


@dataclass(frozen=True)
class DiversityObjectiveHybrid(DiversityObjective):
    """A hybrid objective combines several simple objectives (its terms) by a weighted aggregation, one weight per term.

    Terms are simple objectives only, so each term reads exactly one of the arrays passed to
    `compute`. A hybrid's per-item contributions are its aggregation applied to each item's row of term
    contributions, and its score is the same aggregation applied to the single row of term scores.
    """

    terms: tuple[DiversityObjectiveSimple, ...]
    aggregation: HybridAggregationBase

    @property
    def label(self) -> str:
        """Return e.g. `geomean(MIN_SEPARATION over L2, MIN_SEPARATION over axis 0)`."""
        return self.aggregation.format_label([term.label for term in self.terms])

    def __post_init__(self) -> None:
        """Validate the terms, and the aggregation's weight count against them.

        A one-term hybrid is a `DiversityObjectiveSimple`.

        Raises:
            ValueError: If fewer than 2 terms are given, or the weight count differs from the term count.
            TypeError: If a term is not a simple objective.
        """
        if len(self.terms) < 2:
            raise ValueError(f"A hybrid objective needs at least two terms; got {len(self.terms)}.")
        for term in self.terms:
            if not isinstance(term, DiversityObjectiveSimple):
                raise TypeError(f"A hybrid objective's terms must be simple objectives; got {type(term).__name__}.")
        self.aggregation.validate_term_count(len(self.terms))

    def compute(self, contributions: Sequence[NDArray[np.float32]]) -> float:
        """Return the aggregation of the terms' diversity scores, each term reading its own array."""
        term_scores = np.array(
            [term.compute((contribution,)) for term, contribution in zip(self.terms, contributions, strict=True)],
            dtype=np.float32,
        )
        return self.aggregation.aggregate_scores(term_scores)

    def compute_per_item_contributions(self, contributions: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Return the elementwise aggregation of the terms' arrays, as a fresh float32 array.

        A spec that two terms have enters the aggregation once per term, as `compute` counts a repeated
        term twice; for geometric-mean separation terms under the geometric aggregation, the result gives
        each item exactly the factor by which that item contributes to the objective's score.
        """
        # one row per item, one column per term, so each item's values are contiguous
        stacked = np.stack(contributions, axis=1).astype(np.float32, copy=False)
        return self.aggregation.aggregate_rows(stacked)

    def with_distance_specs(self, replacement_specs: Mapping[DistanceSpec, DistanceSpec]) -> DiversityObjectiveHybrid:
        """Return a copy of this objective with each term's distance spec replaced by its replacement spec."""
        return replace(self, terms=tuple(term.with_distance_specs(replacement_specs) for term in self.terms))

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the aggregation's extra tie-breaker over the terms, if any, then the default tie-breakers.

        The default tie-breakers follow from the terms' diversity metrics, by the rule in the module docstring.
        """
        tie_breakers = super().default_tie_breakers()
        tie_breaker_aggregation_over_terms = self.aggregation.tie_breaker_aggregation_over_terms()
        if tie_breaker_aggregation_over_terms is not None:
            return [DiversityObjectiveHybrid(self.terms, tie_breaker_aggregation_over_terms), *tie_breakers]
        else:
            return tie_breakers

    @cached_property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the terms' specs in term order, a spec repeated once per term that has it."""
        return tuple(term.tracker_spec for term in self.terms)

    @cached_property
    def diversity_metrics(self) -> tuple[DiversityMetric, ...]:
        """Return the terms' diversity metrics in term order."""
        return tuple(term.diversity_metric for term in self.terms)
