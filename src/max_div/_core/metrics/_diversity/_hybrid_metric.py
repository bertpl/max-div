"""The public form of a hybrid diversity metric: several diversity terms aggregated by one mean.

A term pairs a diversity metric with the distance metric it reads (`DiversityTerm`, obtained
through `DiversityMetric.over`); a bare `DiversityMetric` given as a term reads the problem's own
distance. `HybridDiversityMetric` holds the terms and how they are aggregated. Both are immutable
descriptions the problem turns into the objective the solver maximizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._enum import DiversityMetric
from ._objective import DiversityObjectiveHybrid, DiversityObjectiveSimple, HybridObjectiveType

if TYPE_CHECKING:
    from max_div._core.metrics._distance import DistanceMetric


# =================================================================================================
#  DiversityTerm
# =================================================================================================
class DiversityTerm:
    """A diversity metric over one distance metric, one term of a `HybridDiversityMetric`.

    Obtain one through `DiversityMetric.over(distance_metric)`; a term is not constructed directly.
    """

    __slots__ = ("_distance_metric", "_diversity_metric")

    def __init__(self, diversity_metric: DiversityMetric, distance_metric: DistanceMetric) -> None:
        """Pair `diversity_metric` with the `distance_metric` it reads."""
        self._diversity_metric = diversity_metric
        self._distance_metric = distance_metric

    @property
    def diversity_metric(self) -> DiversityMetric:
        """Return the diversity metric of this term."""
        return self._diversity_metric

    @property
    def distance_metric(self) -> DistanceMetric:
        """Return the distance metric this term's diversity metric reads."""
        return self._distance_metric

    @property
    def label(self) -> str:
        """Return a short label, e.g. `MIN_SEPARATION over axis 2`."""
        return f"{self._diversity_metric.value} over {self._distance_metric.label}"

    def _to_objective(self) -> DiversityObjectiveSimple:
        """Return the objective the solver maximizes for this term."""
        return DiversityObjectiveSimple(self._diversity_metric, self._distance_metric)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiversityTerm):
            return NotImplemented
        return self._diversity_metric == other._diversity_metric and self._distance_metric == other._distance_metric

    def __hash__(self) -> int:
        return hash((self._diversity_metric, self._distance_metric))

    def __repr__(self) -> str:
        """Return the call that constructs this term."""
        return f"DiversityMetric.{self._diversity_metric.name}.over({self._distance_metric!r})"


# =================================================================================================
#  HybridDiversityMetric
# =================================================================================================
_AGGREGATION_LABELS = {
    HybridObjectiveType.GEOMETRIC_MEAN: "geomean",
    HybridObjectiveType.ARITHMETIC_MEAN: "mean",
}


class HybridDiversityMetric:
    """Several diversity terms aggregated by their geometric or their arithmetic mean.

    Each term is a `DiversityMetric` over its own distance: a bare `DiversityMetric` reads the
    problem's own distance, and `DiversityMetric.over(distance_metric)` names another one, so one
    solve can spread a selection in the full space and in chosen coordinate projections at once.
    Create instances via `geomean_of` and `mean_of`.

    A hybrid needs at least two terms: a one-term hybrid is the bare metric, and asking for one is
    taken as a mistake. A term may repeat, which weights it once more in the mean. The solver's
    default tie-breakers for a hybrid follow the terms' metrics, whichever mean aggregates them, and
    a hybrid accepts no custom tie-breakers.
    """

    __slots__ = ("_aggregation", "_terms")

    def __init__(self, terms: tuple[DiversityTerm | DiversityMetric, ...], aggregation: HybridObjectiveType) -> None:
        """Validate the terms; use `geomean_of` or `mean_of` to construct a hybrid.

        Raises:
            ValueError: If fewer than two terms are given.
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
        self._terms = terms
        self._aggregation = aggregation

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def geomean_of(cls, *terms: DiversityTerm | DiversityMetric) -> HybridDiversityMetric:
        """Return the geometric mean of the terms' diversity values.

        The geometric mean is zero as soon as one term is zero, so every term must be spread out
        for the selection to score.
        """
        return cls(terms, HybridObjectiveType.GEOMETRIC_MEAN)

    @classmethod
    def mean_of(cls, *terms: DiversityTerm | DiversityMetric) -> HybridDiversityMetric:
        """Return the arithmetic mean of the terms' diversity values.

        A zero term only lowers the arithmetic mean, so a selection can trade a poor spread on one
        term for a better one on another.
        """
        return cls(terms, HybridObjectiveType.ARITHMETIC_MEAN)

    # --------------------------------------------------------------------------
    #  Properties
    # --------------------------------------------------------------------------
    @property
    def terms(self) -> tuple[DiversityTerm | DiversityMetric, ...]:
        """Return the terms as given; a bare `DiversityMetric` reads the problem's own distance."""
        return self._terms

    @property
    def distance_metrics(self) -> tuple[DistanceMetric, ...]:
        """Return the distinct distance metrics the terms name, in first-seen order; bare terms name none."""
        return tuple(dict.fromkeys(term.distance_metric for term in self._terms if isinstance(term, DiversityTerm)))

    @property
    def label(self) -> str:
        """Return a short label, e.g. `geomean(GEOMEAN_SEPARATION, MIN_SEPARATION over axis 0)`."""
        term_labels = ", ".join(term.value if isinstance(term, DiversityMetric) else term.label for term in self._terms)
        return f"{_AGGREGATION_LABELS[self._aggregation]}({term_labels})"

    def _to_objective(self) -> DiversityObjectiveHybrid:
        """Return the objective the solver maximizes for this hybrid."""
        return DiversityObjectiveHybrid(
            tuple(
                DiversityObjectiveSimple(term) if isinstance(term, DiversityMetric) else term._to_objective()
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
        factory = "geomean_of" if self._aggregation == HybridObjectiveType.GEOMETRIC_MEAN else "mean_of"
        term_reprs = ", ".join(
            f"DiversityMetric.{term.name}" if isinstance(term, DiversityMetric) else repr(term) for term in self._terms
        )
        return f"HybridDiversityMetric.{factory}({term_reprs})"
