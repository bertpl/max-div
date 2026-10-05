"""`HybridDiversityMetric` is the public form of a hybrid diversity metric: terms under one weighted aggregation.

A term pairs a diversity metric with the distance metric that it reads (`DiversityTerm`, obtained
through `DiversityMetric.over`), or with none: a bare `DiversityMetric` given as a term becomes a
term whose distance metric is `None`, which reads the problem's own distance.

`HybridDiversityMetric` holds the terms and their aggregation, which carries one weight per term.
`DiversityTerm` and `HybridDiversityMetric` are immutable descriptions; the problem turns them into
the solver's objective.
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
from ._diversity_metric import DiversityMetric
from ._objective import DiversityObjectiveHybrid, DiversityObjectiveSimple

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from max_div._core.metrics._distance import DistanceMetric, DistanceSpec


# ==================================================================================================
#  DiversityTerm
# ==================================================================================================
@dataclass(frozen=True, slots=True)
class DiversityTerm:
    """A diversity term is one term of a `HybridDiversityMetric`: a diversity metric over a distance metric, or none.

    Obtain one that names its distance metric through `DiversityMetric.over(distance_metric)`.
    """

    diversity_metric: DiversityMetric
    # None when the term names no distance metric and reads the problem's own distance
    distance_metric: DistanceMetric | None = None

    @property
    def label(self) -> str:
        """Return a short label, e.g. `MIN_SEPARATION over axis 2`, or `MIN_SEPARATION` without a distance metric."""
        if self.distance_metric is None:
            return self.diversity_metric.label
        else:
            return f"{self.diversity_metric.label} over {self.distance_metric.label}"

    def __repr__(self) -> str:
        """Return the expression for this term that the factory methods of `HybridDiversityMetric` accept."""
        if self.distance_metric is None:
            return repr(self.diversity_metric)
        else:
            return f"{self.diversity_metric!r}.over({self.distance_metric!r})"


# ==================================================================================================
#  HybridDiversityMetric
# ==================================================================================================
class HybridDiversityMetric:
    """A hybrid diversity metric aggregates diversity terms by a weighted geometric mean, arithmetic mean or minimum.

    Each term is a `DiversityMetric` over its own distance: a bare `DiversityMetric` reads the
    problem's own distance, and `DiversityMetric.over(distance_metric)` names another one, so one
    solve can spread a selection in the full space and in chosen coordinate projections at once.
    Create instances via `geomean_of`, `mean_of` and `min_of`.  The hybrid holds every term as a
    `DiversityTerm`: a bare `DiversityMetric` becomes one whose `distance_metric` is `None`.

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
        # a bare DiversityMetric becomes a term that names no distance metric, so the hybrid holds 1 type of term
        self._terms = tuple(DiversityTerm(term) if isinstance(term, DiversityMetric) else term for term in terms)
        self._aggregation = aggregation

    # --------------------------------------------------------------------------
    #  Properties
    # --------------------------------------------------------------------------
    @property
    def terms(self) -> tuple[DiversityTerm, ...]:
        """Return the terms; a bare `DiversityMetric` given as a term is a term whose `distance_metric` is `None`."""
        return self._terms

    @property
    def weights(self) -> tuple[float, ...]:
        """Return one weight per term, in term order; each is 1 unless given."""
        return self._aggregation.weights

    @property
    def named_distance_metrics(self) -> tuple[DistanceMetric, ...]:
        """Return the distinct distance metrics that the terms name, in first-seen order."""
        return tuple(dict.fromkeys(term.distance_metric for term in self._terms if term.distance_metric is not None))

    @property
    def label(self) -> str:
        """Return a short label, e.g. `geomean(GEOMEAN_SEPARATION, MIN_SEPARATION over axis 0)`.

        When any weight differs from 1, all weights follow the terms: `geomean(...; weights 2, 1)`.
        """
        return self._aggregation.format_label([term.label for term in self._terms])

    # --------------------------------------------------------------------------
    #  Conversion to an objective
    # --------------------------------------------------------------------------
    def _to_objective(
        self, distance_spec_of: Callable[[DistanceMetric | None], DistanceSpec]
    ) -> DiversityObjectiveHybrid:
        """Return the objective that the solver maximizes for this hybrid, each term over the distances that it reads.

        The hybrid passes each term's `distance_metric` to `distance_spec_of` without inspecting it, `None`
        included, so the problem alone decides what a term without a distance metric reads.

        Args:
            distance_spec_of: the problem's `_distance_spec_of`, which returns the distance spec of a
                term's distance metric.
        """
        return DiversityObjectiveHybrid(
            tuple(
                DiversityObjectiveSimple(term.diversity_metric, distance_spec_of(term.distance_metric))
                for term in self._terms
            ),
            self._aggregation,
        )

    # --------------------------------------------------------------------------
    #  Representation
    # --------------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        """Return whether `other` is a hybrid with equal terms and an equal aggregation."""
        if not isinstance(other, HybridDiversityMetric):
            return NotImplemented
        else:
            return self._terms == other._terms and self._aggregation == other._aggregation

    def __hash__(self) -> int:
        """Return a hash of the terms and the aggregation, consistent with `__eq__`."""
        return hash((self._terms, self._aggregation))

    def __repr__(self) -> str:
        """Return the factory call that constructs this hybrid."""
        term_reprs = ", ".join(repr(term) for term in self._terms)
        weights_suffix = f", weights={self._aggregation.weights!r}" if self._aggregation.has_non_unit_weights else ""
        return f"HybridDiversityMetric.{self._aggregation.name}_of({term_reprs}{weights_suffix})"

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
