"""`HybridDiversityMetric` is the public form of a hybrid diversity metric: terms under one weighted aggregation.

A term pairs a diversity metric with the distance metric it reads (`DiversityTerm`, obtained
through `DiversityMetric.over`); a bare `DiversityMetric` given as a term reads the problem's own
distance. `HybridDiversityMetric` holds the terms and their aggregation, which carries one weight per
term. `DiversityTerm` and `HybridDiversityMetric` are immutable descriptions; the problem turns them into the
solver's objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._aggregation import (
    HybridAggregationArithmeticMean,
    HybridAggregationBase,
    HybridAggregationGeometricMean,
    HybridAggregationMinimum,
)
from ._enum import DiversityMetric
from ._objective import DiversityObjectiveHybrid, DiversityObjectiveSimple

if TYPE_CHECKING:
    from collections.abc import Sequence

    from max_div._core.metrics._distance import DistanceMetric


# ==================================================================================================
#  DiversityTerm
# ==================================================================================================
@dataclass(frozen=True, slots=True)
class DiversityTerm:
    """A diversity metric over one distance metric, one term of a `HybridDiversityMetric`.

    Obtain one through `DiversityMetric.over(distance_metric)`; a term is not constructed directly.
    """

    diversity_metric: DiversityMetric
    distance_metric: DistanceMetric

    @property
    def label(self) -> str:
        """Return a short label, e.g. `MIN_SEPARATION over axis 2`."""
        return f"{self.diversity_metric.value} over {self.distance_metric.label}"

    def __repr__(self) -> str:
        """Return the call that constructs this term."""
        return f"DiversityMetric.{self.diversity_metric.name}.over({self.distance_metric!r})"


# ==================================================================================================
#  HybridDiversityMetric
# ==================================================================================================
class HybridDiversityMetric:
    """A hybrid diversity metric aggregates diversity terms by a weighted geometric mean, arithmetic mean or minimum.

    Each term is a `DiversityMetric` over its own distance: a bare `DiversityMetric` reads the
    problem's own distance, and `DiversityMetric.over(distance_metric)` names another one, so one
    solve can spread a selection in the full space and in chosen coordinate projections at once.
    Create instances via `geomean_of`, `mean_of` and `min_of`.

    Each term carries a weight, 1 unless given, whose meaning depends on the aggregation: an exponent
    on the term's value in the geometric mean, a multiplier of the term's value in the arithmetic mean
    and in the minimum.

    A hybrid needs at least 2 terms: a one-term hybrid is the bare metric, and asking for one is taken
    as a mistake. A term may repeat, which counts it once more in a mean; in the minimum, only the
    copy with the smallest weight has an effect.

    The solver's default tie-breakers for a hybrid follow the terms' metrics, whichever aggregation
    combines them; a `min_of` hybrid first gets the geomean of its terms. A hybrid accepts no custom
    tie-breakers.
    """

    __slots__ = ("_aggregation", "_terms")

    def __init__(self, terms: tuple[DiversityTerm | DiversityMetric, ...], aggregation: HybridAggregationBase) -> None:
        """Validate the terms against the aggregation; use `geomean_of`, `mean_of` or `min_of` to construct a hybrid.

        Raises:
            ValueError: If fewer than 2 terms are given, or the aggregation holds a different number
                of weights.
            TypeError: If a term is not a `DiversityMetric` or a `DiversityTerm`; a hybrid does not nest.
        """
        if len(terms) < 2:
            raise ValueError(f"A hybrid diversity metric needs at least two terms; got {len(terms)}.")
        for term in terms:
            if not isinstance(term, (DiversityMetric, DiversityTerm)):
                raise TypeError(
                    "Each term of a hybrid diversity metric must be a DiversityMetric or a "
                    f"DiversityMetric.over(...); got {type(term).__name__}."
                )
        aggregation.validate_term_count(len(terms))
        self._terms = terms
        self._aggregation = aggregation

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def geomean_of(
        cls, *terms: DiversityTerm | DiversityMetric, weights: Sequence[float] | None = None
    ) -> HybridDiversityMetric:
        """Return the weighted geometric mean of the terms' diversity values.

        The geometric mean is zero as soon as one term is zero, so every term must be spread out
        for the selection to score. Rescaling a term by a constant does not change which selection
        scores highest. A term's weight is its exponent: the result is
        `(prod_t value_t ** weight_t) ** (1 / sum(weights))`, the plain geometric mean when every
        weight is equal.

        Args:
            terms: the diversity terms, each a `DiversityMetric` or a `DiversityMetric.over(...)`.
            weights: one positive, finite weight per term, in term order; `None` weights every term 1.

        Raises:
            ValueError: If fewer than 2 terms are given, or a weight is missing, extra, or not a
                positive, finite number.
        """
        return cls(terms, HybridAggregationGeometricMean.from_weights(weights, len(terms)))

    @classmethod
    def mean_of(
        cls, *terms: DiversityTerm | DiversityMetric, weights: Sequence[float] | None = None
    ) -> HybridDiversityMetric:
        """Return the weighted arithmetic mean of the terms' diversity values.

        A zero term only lowers the arithmetic mean, so a selection can trade a poor spread on one
        term for a better one on another. Each term enters the mean at its own scale, so a term over a
        larger distance counts for more unless its weight compensates. The result is
        `sum_t weight_t * value_t / sum(weights)`, the plain arithmetic mean when every weight is equal.

        Args:
            terms: the diversity terms, each a `DiversityMetric` or a `DiversityMetric.over(...)`.
            weights: one positive, finite weight per term, in term order; `None` weights every term 1.

        Raises:
            ValueError: If fewer than 2 terms are given, or a weight is missing, extra, or not a
                positive, finite number.
        """
        return cls(terms, HybridAggregationArithmeticMean.from_weights(weights, len(terms)))

    @classmethod
    def min_of(
        cls, *terms: DiversityTerm | DiversityMetric, weights: Sequence[float] | None = None
    ) -> HybridDiversityMetric:
        """Return the smallest of the terms' diversity values, each multiplied by its weight.

        The score is the lowest of the weighted terms, so a high value in one term cannot compensate
        for a low value in another.

        The minimum compares raw values, so terms of different scales need
        weights that bring them onto a common one: with k well-spread points in the unit square, the
        min separation over L2 shrinks like 1/sqrt(k) and along one axis like 1/k, so weights sqrt(k)
        and k make them comparable.

        The weights are not normalized.

        Args:
            terms: the diversity terms, each a `DiversityMetric` or a `DiversityMetric.over(...)`.
            weights: one positive, finite weight per term, in term order; `None` weights every term 1.

        Raises:
            ValueError: If fewer than 2 terms are given, or a weight is missing, extra, or not a
                positive, finite number.
        """
        return cls(terms, HybridAggregationMinimum.from_weights(weights, len(terms)))

    # --------------------------------------------------------------------------
    #  Properties
    # --------------------------------------------------------------------------
    @property
    def terms(self) -> tuple[DiversityTerm | DiversityMetric, ...]:
        """Return the terms as given; a bare `DiversityMetric` reads the problem's own distance."""
        return self._terms

    @property
    def weights(self) -> tuple[float, ...]:
        """Return one weight per term, in term order; each is 1 unless given."""
        return self._aggregation.weights

    @property
    def named_distance_metrics(self) -> tuple[DistanceMetric, ...]:
        """Return the distinct distance metrics the terms name, in first-seen order; bare terms name none."""
        return tuple(dict.fromkeys(term.distance_metric for term in self._terms if isinstance(term, DiversityTerm)))

    @property
    def label(self) -> str:
        """Return a short label, e.g. `geomean(GEOMEAN_SEPARATION, MIN_SEPARATION over axis 0)`.

        When any weight differs from 1, all weights follow the terms: `geomean(...; weights 2, 1)`.
        """
        return self._to_objective().label

    def _to_objective(self, bare_term_distance_metric: DistanceMetric | None = None) -> DiversityObjectiveHybrid:
        """Return the solver's objective for this hybrid.

        Args:
            bare_term_distance_metric: the distance metric that each bare term reads, a bare term
                being a plain `DiversityMetric` given without `over`.  Pass None for a distance-input
                problem, whose given distances have no metric.
        """
        return DiversityObjectiveHybrid(
            tuple(
                DiversityObjectiveSimple(term, bare_term_distance_metric)
                if isinstance(term, DiversityMetric)
                else DiversityObjectiveSimple(term.diversity_metric, term.distance_metric)
                for term in self._terms
            ),
            self._aggregation,
        )

    # --------------------------------------------------------------------------
    #  Representation
    # --------------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HybridDiversityMetric):
            return NotImplemented
        return self._terms == other._terms and self._aggregation == other._aggregation

    def __hash__(self) -> int:
        return hash((self._terms, self._aggregation))

    def __repr__(self) -> str:
        """Return the factory call that constructs this hybrid."""
        term_reprs = ", ".join(
            f"DiversityMetric.{term.name}" if isinstance(term, DiversityMetric) else repr(term) for term in self._terms
        )
        weights_suffix = f", weights={self._aggregation.weights!r}" if self._aggregation.has_non_unit_weights else ""
        return f"HybridDiversityMetric.{self._aggregation.name}_of({term_reprs}{weights_suffix})"
