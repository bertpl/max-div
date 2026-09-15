"""A diversity objective is what the solver maximizes; a tie-breaker is a further objective the solver ranks ties by.

A `DiversityObjective` is one of two kinds, each holding only the fields that kind of objective needs:

- `DiversityObjectiveSimple` — one diversity metric over one distance metric.
- `DiversityObjectiveHybrid` — several simple objectives (its terms) aggregated by a geometric or an
  arithmetic mean.

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

Each tie-breaker is computed over every distinct distance metric of the objective, as a hybrid when there are several.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from max_div._core._math.geomean import geomean_f32, geomean_per_row_f32

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

        The rule in the module docstring picks the tie-breaker metrics from this objective's
        diversity metrics.
        """
        distance_metrics = self.distinct_distance_metrics()
        return [
            _objective_over(spec.metric, spec.aggregation, distance_metrics)
            for spec in _tie_breaker_specs_for(self.diversity_metrics)
        ]

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

    def compute_per_item_contributions(self, contributions: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Return the one array unchanged: a single spec's contribution is the objective's own contribution."""
        return contributions[0]

    @cached_property
    def tracker_spec(self) -> DiversityTrackerSpec:
        """Return the one spec of this objective: the single entry of `tracker_specs`, as a shorthand."""
        return DiversityTrackerSpec(self.distance_metric, self.diversity_metric.contribution_family)

    @cached_property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the one spec of this objective."""
        return (self.tracker_spec,)

    @cached_property
    def diversity_metrics(self) -> tuple[DiversityMetric, ...]:
        """Return the one diversity metric of this objective."""
        return (self.diversity_metric,)


class HybridObjectiveType(StrEnum):
    """How a hybrid objective aggregates its terms: by their geometric mean or by their arithmetic mean."""

    GEOMETRIC_MEAN = "GEOMETRIC_MEAN"
    ARITHMETIC_MEAN = "ARITHMETIC_MEAN"


@dataclass(frozen=True)
class DiversityObjectiveHybrid(DiversityObjective):
    """Several simple objectives (its terms) aggregated by their geometric or arithmetic mean.

    Terms are simple objectives only, so each term reads exactly one of the arrays passed to
    `compute`. The solver maximizes a hybrid with the geometric aggregation.
    """

    terms: tuple[DiversityObjectiveSimple, ...]
    aggregation: HybridObjectiveType = HybridObjectiveType.GEOMETRIC_MEAN

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
        """Return the aggregation of the terms' diversity scores, each term reading its own array."""
        term_scores = np.array(
            [term.compute((contribution,)) for term, contribution in zip(self.terms, contributions, strict=True)],
            dtype=np.float32,
        )
        if self.aggregation == HybridObjectiveType.GEOMETRIC_MEAN:
            return float(geomean_f32(term_scores))
        else:
            return float(np.mean(term_scores))

    def compute_per_item_contributions(self, contributions: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Return the elementwise aggregation of the terms' arrays, as a fresh float32 array.

        A spec that two terms have enters the aggregation once per term, as `compute` counts a repeated
        term twice; for geometric-mean separation terms under the geometric aggregation, the result gives
        each item exactly the factor by which that item contributes to the objective's score.
        """
        # one row per item, one column per term, so each item's values are contiguous
        stacked = np.stack(contributions, axis=1).astype(np.float32, copy=False)
        if self.aggregation == HybridObjectiveType.GEOMETRIC_MEAN:
            aggregated = np.empty(stacked.shape[0], dtype=np.float32)
            geomean_per_row_f32(stacked, aggregated)
            return aggregated
        else:
            return stacked.mean(axis=1, dtype=np.float32)

    @cached_property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """Return the terms' specs in term order, a spec repeated once per term that has it."""
        return tuple(term.tracker_spec for term in self.terms)

    @cached_property
    def diversity_metrics(self) -> tuple[DiversityMetric, ...]:
        """Return the terms' diversity metrics in term order."""
        return tuple(term.diversity_metric for term in self.terms)


# =================================================================================================
#  Helpers
# =================================================================================================
class _TieBreakerSpec(NamedTuple):
    """Pairs a tie-breaker diversity metric with the aggregation its hybrid form uses over several distances."""

    metric: DiversityMetric
    aggregation: HybridObjectiveType


# the approximate geomean is higher for a more uniform spread, which increases the separation of the
# closest pair; aggregated geometrically, a zero on one distance lowers it without making the
# tie-breaker zero
_APPROX_GEOMEAN_TIE_BREAKER_SPEC = _TieBreakerSpec(
    DiversityMetric.APPROX_GEOMEAN_SEPARATION, HybridObjectiveType.GEOMETRIC_MEAN
)
# the non-zero fraction rises as the number of coincident pairs falls, even once the geomean is
# stuck at zero; aggregated arithmetically, a distance with no non-zero separation only lowers it,
# so fixing another distance still ranks higher
_NON_ZERO_FRAC_TIE_BREAKER_SPEC = _TieBreakerSpec(
    DiversityMetric.NON_ZERO_SEPARATION_FRAC, HybridObjectiveType.ARITHMETIC_MEAN
)

# one coincident pair pins each of these separation aggregates at zero
_ZERO_PINNED_METRICS = frozenset(
    {
        DiversityMetric.GEOMEAN_SEPARATION,
        DiversityMetric.APPROX_GEOMEAN_SEPARATION,
        DiversityMetric.HARMONIC_MEAN_SEPARATION,
    }
)


def _tie_breaker_specs_for(diversity_metrics: tuple[DiversityMetric, ...]) -> tuple[_TieBreakerSpec, ...]:
    """Return the tie-breaker specs that the given term metrics call for, in ranking order; empty when none does."""
    # min-separation reacts only to the closest pair, so every swap that leaves that pair alone ties
    needs_approx_geomean_tie_breaker = DiversityMetric.MIN_SEPARATION in diversity_metrics
    # once more than one pair coincides, single swaps cannot raise a zero-pinned metric above zero;
    # a min-separation objective needs the non-zero fraction too, for when the approximate geomean
    # has underflowed to zero
    needs_non_zero_frac_tie_breaker = needs_approx_geomean_tie_breaker or any(
        metric in _ZERO_PINNED_METRICS for metric in diversity_metrics
    )
    specs = []
    if needs_approx_geomean_tie_breaker:
        specs.append(_APPROX_GEOMEAN_TIE_BREAKER_SPEC)
    if needs_non_zero_frac_tie_breaker:
        specs.append(_NON_ZERO_FRAC_TIE_BREAKER_SPEC)
    return tuple(specs)


def _objective_over(
    diversity_metric: DiversityMetric,
    aggregation: HybridObjectiveType,
    distance_metrics: tuple[DistanceMetric | None, ...],
) -> DiversityObjective:
    """Return `diversity_metric` over the given distance metrics: simple over one, a hybrid over several."""
    terms = tuple(DiversityObjectiveSimple(diversity_metric, distance_metric) for distance_metric in distance_metrics)
    if len(terms) == 1:
        return terms[0]
    else:
        return DiversityObjectiveHybrid(terms, aggregation)
