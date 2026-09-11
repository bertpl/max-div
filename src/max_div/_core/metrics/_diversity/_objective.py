"""A diversity objective is what the solver maximizes: one or more diversity terms.

The solver, its config, presets and strategies read this type, never the bare `DiversityMetric`
enum, because an objective of several terms cannot be a single enum member; the terms live here
and each consumer reads the derived facts it needs off the objective.
"""

from dataclasses import dataclass

from ._enum import DiversityContributionFamily, DiversityMetric
from ._term import DiversityTerm


@dataclass(frozen=True)
class DiversityObjective:
    """The diversity objective the solver maximizes, as a tuple of terms.

    The objective is built from a problem's `diversity_terms`; its accessors expose the derived
    facts each consumer reads off the terms.
    """

    terms: tuple[DiversityTerm, ...]

    def __post_init__(self) -> None:
        """Reject an objective with no terms; there is nothing to maximize."""
        if not self.terms:
            raise ValueError("A diversity objective needs at least one term.")

    @property
    def main_metric(self) -> DiversityMetric:
        """Return the single term's metric.

        Defined only for single-term objectives.
        """
        return self.terms[0].metric

    @property
    def main_family(self) -> DiversityContributionFamily:
        """Return the contribution family the single term consumes."""
        return self.main_metric.contribution_family

    @property
    def contribution_families(self) -> tuple[DiversityContributionFamily, ...]:
        """Return the distinct contribution families the objective's terms consume, in first-seen order."""
        return tuple(dict.fromkeys(term.metric.contribution_family for term in self.terms))

    @property
    def has_single_separation_tracker(self) -> bool:
        """Return whether one separation tracker serves the whole objective.

        True for one term in the separation family — the case the batched farthest-point
        initialization is tailored to.
        """
        return len(self.terms) == 1 and self.main_family == DiversityContributionFamily.SEPARATION

    @property
    def default_tie_breakers(self) -> list[DiversityMetric]:
        """Return the tie-breakers to score with when the caller sets none, chosen from the main metric.

        A near-degenerate main metric, where many selections share a score, gets tie-breakers that
        separate them; every other metric gets none.
        """
        metric = self.main_metric
        if metric == DiversityMetric.MIN_SEPARATION:
            return [DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        if metric in (DiversityMetric.GEOMEAN_SEPARATION, DiversityMetric.APPROX_GEOMEAN_SEPARATION):
            return [DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        return []
