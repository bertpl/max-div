"""A diversity objective is what the solver maximizes: one or more diversity terms.

The solver, its config, presets and strategies read this type, never the bare `DiversityMetric`
enum, because an objective of several terms cannot be a single enum member; the terms live here.
A tie-breaker is an objective too, of the flattened kind, so the score generator evaluates
objectives and nothing else.
"""

from dataclasses import dataclass
from enum import StrEnum

from max_div._core.metrics._distance import DistanceMetric

from ._enum import DiversityContributionFamily, DiversityMetric
from ._term import DiversityTerm

# One row of per-item contributions the solver tracks: a contribution family over a distance, `None`
# standing for the problem's given distance.  Terms that share a key share the row.
ContributionKey = tuple[DiversityContributionFamily, DistanceMetric | None]


class TermAggregation(StrEnum):
    """Enum for how a diversity objective combines its terms into one value.

    Members
    -------

        - HIERARCHICAL:  each term's metric reduces that term's row, and the objective is the geometric
                         mean of the term values; the kind of the primary objective
        - FLATTENED:     one metric reduces all the terms' rows concatenated into one list, so every
                         term carries that same metric; the kind of a tie-breaker
    """

    HIERARCHICAL = "HIERARCHICAL"
    FLATTENED = "FLATTENED"


@dataclass(frozen=True)
class DiversityObjective:
    """The diversity objective the solver maximizes, as a tuple of terms and how they aggregate.

    The primary objective is built from a problem's `diversity_terms` and aggregates hierarchically;
    a tie-breaker is built by `tie_breaker` and aggregates flattened. The accessors expose the
    derived facts each consumer reads off the terms.
    """

    terms: tuple[DiversityTerm, ...]
    aggregation: TermAggregation = TermAggregation.HIERARCHICAL

    def __post_init__(self) -> None:
        """Reject an objective with no terms, and a flattened one whose terms carry different metrics."""
        if not self.terms:
            raise ValueError("A diversity objective needs at least one term.")
        if self.aggregation == TermAggregation.FLATTENED and len({term.diversity_metric for term in self.terms}) > 1:
            raise ValueError("A flattened objective reduces with one metric, so all its terms must carry that metric.")

    # --------------------------------------------------------------------------
    #  Single-term accessors
    # --------------------------------------------------------------------------
    @property
    def main_diversity_metric(self) -> DiversityMetric:
        """Return the single term's diversity metric.

        Defined only for single-term objectives. The `main_*` accessors are interim, for the
        single-term case; remove them once an objective can hold several terms.
        """
        return self.terms[0].diversity_metric

    @property
    def main_contribution_family(self) -> DiversityContributionFamily:
        """Return the contribution family the single term consumes."""
        return self.main_diversity_metric.contribution_family

    @property
    def has_single_separation_term(self) -> bool:
        """Return whether the objective is a single term in the separation family."""
        return len(self.terms) == 1 and self.main_contribution_family == DiversityContributionFamily.SEPARATION

    # --------------------------------------------------------------------------
    #  Derived facts
    # --------------------------------------------------------------------------
    @property
    def contribution_keys(self) -> tuple[ContributionKey, ...]:
        """Return the distinct (contribution family, distance) keys the terms read, in first-seen order."""
        return tuple(
            dict.fromkeys((term.diversity_metric.contribution_family, term.distance_metric) for term in self.terms)
        )

    # --------------------------------------------------------------------------
    #  Tie-breakers
    # --------------------------------------------------------------------------
    def tie_breaker(self, diversity_metric: DiversityMetric) -> "DiversityObjective":
        """Return the flattened objective of `diversity_metric` over each distance this objective measures over.

        The distances are those of this objective's terms, distinct and in first-seen order, so the
        tie-breaker reads the rows the primary already tracks when the families match, and brings
        its own rows when they do not.
        """
        distances = dict.fromkeys(term.distance_metric for term in self.terms)
        return DiversityObjective(
            terms=tuple(DiversityTerm(diversity_metric, distance) for distance in distances),
            aggregation=TermAggregation.FLATTENED,
        )

    @property
    def default_tie_breakers(self) -> list["DiversityObjective"]:
        """Return the tie-breakers to score with when the caller sets none, determined by the main metric.

        A near-degenerate main metric, where many selections share a score, gets tie-breakers that
        separate them; every other metric gets none. Each is built by `tie_breaker`.
        """
        metric = self.main_diversity_metric
        if metric == DiversityMetric.MIN_SEPARATION:
            metrics = [DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        elif metric in (DiversityMetric.GEOMEAN_SEPARATION, DiversityMetric.APPROX_GEOMEAN_SEPARATION):
            metrics = [DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        else:
            metrics = []
        return [self.tie_breaker(tie_breaker_metric) for tie_breaker_metric in metrics]
