"""A diversity objective is what the solver maximizes: one or more diversity terms.

The solver, its config, presets and strategies read this type, never the bare `DiversityMetric`
enum — a hybrid objective cannot be one enum member, so the terms live here and the derived facts
each consumer asks for are read off the objective. The objective holds one term today; the hybrid
adds more.
"""

from dataclasses import dataclass

from ._enum import DiversityContributionFamily, DiversityMetric
from ._term import DiversityTerm


@dataclass(frozen=True)
class DiversityObjective:
    """The diversity objective the solver maximizes, as a tuple of terms.

    Built from a problem's `diversity_terms`. The accessors expose the facts consumers read: the
    single term's metric and contribution family, the contribution families the objective needs,
    the default tie-breakers, and whether one separation tracker serves the whole objective.
    """

    terms: tuple[DiversityTerm, ...]

    def __post_init__(self) -> None:
        """Reject an objective with no terms; there is nothing to maximize."""
        if not self.terms:
            raise ValueError("A diversity objective needs at least one term.")

    @property
    def main_metric(self) -> DiversityMetric:
        """Return the single term's metric.

        Defined while the objective is single-term; the multi-term hybrid replaces the consumers
        that read it (scoring and tracking) rather than this accessor gaining a meaning for many
        terms.
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
    def uses_single_separation_tracker(self) -> bool:
        """Return whether one separation tracker serves the whole objective.

        True for one term in the separation family — the case the batched farthest-point
        initialization is tailored to.
        """
        return len(self.terms) == 1 and self.main_family == DiversityContributionFamily.SEPARATION

    @property
    def default_tie_breakers(self) -> list[DiversityMetric]:
        """Return the tie-breakers to score with when the caller sets none, chosen from the main metric.

        A near-degenerate main metric (minimum or geometric-mean separation, where many selections
        share a score) gets tie-breakers that separate them; every other metric gets none.
        """
        metric = self.main_metric
        if metric == DiversityMetric.MIN_SEPARATION:
            return [DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        if metric in (DiversityMetric.GEOMEAN_SEPARATION, DiversityMetric.APPROX_GEOMEAN_SEPARATION):
            return [DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        return []
