"""A diversity objective is what the solver maximizes: one or more diversity terms.

The solver maximizes one objective, the primary objective, and ranks selections that tie on it by
tie-breaker objectives. The solver, its config, presets and strategies read this type, never the
bare `DiversityMetric` enum, because an objective of several terms cannot be a single enum member;
the terms live here. A tie-breaker is an objective too (see `TermAggregationType`), so the solver,
its config and the builders carry one type for both; the solver state still reads each objective's
bare metric.
"""

from dataclasses import dataclass

from max_div._core.metrics._distance import DistanceMetric

from ._enum import DiversityContributionFamily, DiversityMetric, TermAggregationType
from ._term import DiversityTerm

# The (contribution family, distance) pair that a term reads its per-item contributions from; the
# distance is the term's, as `DiversityTerm` defines it.  Terms with equal keys read the same
# contributions.
DiversityContributionKey = tuple[DiversityContributionFamily, DistanceMetric | None]


@dataclass(frozen=True)
class DiversityObjective:
    """The diversity objective the solver maximizes: a tuple of terms plus the rule that combines their values.

    The primary objective is built from a problem's `diversity_terms` and has `GEOMEAN_OF_TERMS`
    aggregation; a tie-breaker is built by `build_tie_breaker` and has `FLATTENED_TERMS` aggregation.
    """

    terms: tuple[DiversityTerm, ...]
    aggregation_type: TermAggregationType = TermAggregationType.GEOMEAN_OF_TERMS

    def __post_init__(self) -> None:
        """Reject an objective with no terms, and a flattened one whose terms carry different metrics."""
        if not self.terms:
            raise ValueError("A diversity objective needs at least one term.")
        metrics = {term.diversity_metric for term in self.terms}
        if self.aggregation_type == TermAggregationType.FLATTENED_TERMS and len(metrics) > 1:
            raise ValueError(
                "A FLATTENED_TERMS objective is computed with one diversity metric, "
                "so all its terms must use the same metric."
            )

    # --------------------------------------------------------------------------
    #  Single-term accessors
    # --------------------------------------------------------------------------
    @property
    def main_diversity_metric(self) -> DiversityMetric:
        """Return the single term's diversity metric.

        Defined only for single-term objectives: the `main_*` accessors assume the one term that
        `MaxDivProblem.diversity_terms` yields, and go once a problem can hold several.
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
    def contribution_keys(self) -> tuple[DiversityContributionKey, ...]:
        """Return the distinct contribution keys that the terms read, in first-seen order."""
        return tuple(
            dict.fromkeys((term.diversity_metric.contribution_family, term.distance_metric) for term in self.terms)
        )

    # --------------------------------------------------------------------------
    #  Tie-breakers
    # --------------------------------------------------------------------------
    def build_tie_breaker(self, tie_breaker_metric: DiversityMetric) -> "DiversityObjective":
        """Return the flattened objective with one term per distinct distance of this objective's terms.

        Every term carries `tie_breaker_metric`; the distances keep the order in which the terms
        first use them.
        """
        distance_metrics = dict.fromkeys(term.distance_metric for term in self.terms)
        return DiversityObjective(
            terms=tuple(DiversityTerm(tie_breaker_metric, distance_metric) for distance_metric in distance_metrics),
            aggregation_type=TermAggregationType.FLATTENED_TERMS,
        )

    @property
    def default_tie_breakers(self) -> list["DiversityObjective"]:
        """Return the tie-breakers to score with when the caller sets none, determined by the main metric.

        A near-degenerate main metric, where many selections share a score, gets tie-breakers that
        separate them; every other metric gets none.
        """
        metric = self.main_diversity_metric
        if metric == DiversityMetric.MIN_SEPARATION:
            metrics = [DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        elif metric in (DiversityMetric.GEOMEAN_SEPARATION, DiversityMetric.APPROX_GEOMEAN_SEPARATION):
            metrics = [DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        else:
            metrics = []
        return [self.build_tie_breaker(tie_breaker_metric) for tie_breaker_metric in metrics]
