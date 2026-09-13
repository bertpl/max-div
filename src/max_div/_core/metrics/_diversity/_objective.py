"""A diversity objective is what the solver maximizes; a tie-breaker is a further objective the solver ranks ties by.

A `DiversityObjective` is one of three kinds, each holding only the fields that kind of objective needs:

- `DiversityObjectiveSimple` — one diversity metric over one distance metric.
- `DiversityObjectiveHybridGeoMean` — the geometric mean of several simpler objectives (its terms).
- `DiversityObjectiveHybridFlattened` — one diversity metric over several distance metrics at once,
  read as one joined input; a hybrid objective's tie-breakers take this shape.

Every kind computes its own diversity score (`compute`) from the per-item contributions the solver
tracks. The solver passes `compute` one array per spec of `tracker_specs`, in that order. The solver,
its config, builders, presets and strategies read this type, never a bare `DiversityMetric`, because
an objective of several terms is not a single diversity metric.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from max_div._core._math.geomean import geomean_f32

from ._enum import DiversityContributionFamily, DiversityMetric

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

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
    specs a subclass declares, the base derives the facts consumers read: the distinct distance
    metrics, and whether one separation tracker serves the objective.
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
        """The distinct specs this objective reads, in first-seen order (one tracker each)."""

    @abstractmethod
    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the tie-breaker objectives to rank ties by when the caller sets none of its own."""

    def distinct_distance_metrics(self) -> tuple[DistanceMetric | None, ...]:
        """Return the distinct distance metrics this objective reads, in first-seen order."""
        return tuple(dict.fromkeys(spec.distance_metric for spec in self.tracker_specs))

    def has_single_separation_tracker(self) -> bool:
        """Return whether one separation tracker serves this objective, which the batched-init fast path requires.

        True when the objective reads exactly one spec, of the separation family.
        """
        specs = self.tracker_specs
        return len(specs) == 1 and specs[0].contribution_family == DiversityContributionFamily.SEPARATION


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
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """The one spec this objective reads."""
        return (self._spec,)

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the separating tie-breakers a near-degenerate diversity metric needs; other metrics get none.

        A near-degenerate diversity metric, where many selections share a score, gets tie-breakers
        that separate them, each over this objective's own distance metric.
        """
        return [
            DiversityObjectiveSimple(tie_breaker_metric, self.distance_metric)
            for tie_breaker_metric in _separating_tie_breaker_metrics(self.diversity_metric)
        ]

    @cached_property
    def _spec(self) -> DiversityTrackerSpec:
        """This objective's (distance metric, contribution family) spec."""
        return DiversityTrackerSpec(self.distance_metric, self.diversity_metric.contribution_family)


@dataclass(frozen=True)
class DiversityObjectiveHybridGeoMean(DiversityObjective):
    """The geometric mean of several diversity terms, each a simple objective over its own distance metric.

    Terms are simple objectives only, so each term reads exactly one of the contribution arrays
    passed to `compute`.
    """

    terms: tuple[DiversityObjectiveSimple, ...]

    def __post_init__(self) -> None:
        """Reject fewer than two terms and any term that is not a simple objective.

        A one-term geometric mean is a `DiversityObjectiveSimple`.
        """
        if len(self.terms) < 2:
            raise ValueError(f"A geometric-mean hybrid needs at least two terms; got {len(self.terms)}.")
        for term in self.terms:
            if not isinstance(term, DiversityObjectiveSimple):
                raise TypeError(
                    f"A geometric-mean hybrid's terms must be simple objectives; got {type(term).__name__}."
                )

    def compute(self, contributions: Sequence[NDArray[np.float32]]) -> float:
        """Return the geometric mean of the terms' diversity scores."""
        term_scores = np.array([score_term(contributions) for score_term in self._term_scorers], dtype=np.float32)
        return float(geomean_f32(term_scores))

    @cached_property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """The distinct specs of the terms, in first-seen order."""
        return distinct_tracker_specs_of(self.terms)

    @cached_property
    def _term_scorers(self) -> tuple[Callable[[Sequence[NDArray[np.float32]]], float], ...]:
        """One scorer per term, each picking its term's one array out of the arrays passed to `compute`.

        Two terms that read the same spec get the same array.
        """
        return tuple(objective_scorer(term, self.tracker_specs) for term in self.terms)

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return the separating tie-breakers over the distinct distance metrics the terms read."""
        distances = self.distinct_distance_metrics()
        tie_breaker_metrics = (DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC)
        return [DiversityObjectiveHybridFlattened(metric, distances) for metric in tie_breaker_metrics]


@dataclass(frozen=True)
class DiversityObjectiveHybridFlattened(DiversityObjective):
    """One diversity metric over several distance metrics at once: it reads their contributions as one joined input.

    A hybrid objective's tie-breakers take this shape (one metric over the hybrid's distances). Holds
    one diversity metric by construction; a distance metric is `None` for the problem's own distance.
    """

    diversity_metric: DiversityMetric
    distance_metrics: tuple[DistanceMetric | None, ...]

    def compute(self, contributions: Sequence[NDArray[np.float32]]) -> float:
        """Reduce the joined contribution arrays of all this objective's specs with its diversity metric."""
        return float(self.diversity_metric.compute(np.concatenate(contributions)))

    @cached_property
    def tracker_specs(self) -> tuple[DiversityTrackerSpec, ...]:
        """One spec per distinct distance metric, all of the diversity metric's family."""
        family = self.diversity_metric.contribution_family
        return tuple(
            dict.fromkeys(DiversityTrackerSpec(distance_metric, family) for distance_metric in self.distance_metrics)
        )

    def default_tie_breakers(self) -> list[DiversityObjective]:
        """Return none: a tie-breaker is not itself ranked by further tie-breakers."""
        return []


# =================================================================================================
#  Helpers
# =================================================================================================
def distinct_tracker_specs_of(objectives: Iterable[DiversityObjective]) -> tuple[DiversityTrackerSpec, ...]:
    """Return the distinct specs the objectives read, in the order the objectives list them.

    Each objective contributes its `tracker_specs` in its own order, and a repeated spec keeps its
    first position.
    """
    return tuple(dict.fromkeys(spec for objective in objectives for spec in objective.tracker_specs))


def objective_scorer(
    diversity_objective: DiversityObjective, tracked_specs: tuple[DiversityTrackerSpec, ...]
) -> Callable[[Sequence[NDArray[np.float32]]], float]:
    """Return the function that scores `diversity_objective` from the tracked arrays, ordered as `tracked_specs`.

    `tracked_specs` must contain every spec in `diversity_objective.tracker_specs`. The objective's
    `compute` reads its arrays in its own spec order, so the scorer picks them out of the tracked
    arrays by position. When the objective's specs are exactly the tracked specs, in that order, as
    in the usual single-metric problem where every objective reads the same spec, `compute` itself
    is the scorer, with no picking and no extra call on the hot path.
    """
    positions = tuple(tracked_specs.index(spec) for spec in diversity_objective.tracker_specs)
    compute_objective_score = diversity_objective.compute
    if positions == tuple(range(len(tracked_specs))):
        return compute_objective_score
    elif len(positions) == 1:
        # `itemgetter` is not used here: with one position it returns the array itself, and `compute`
        # takes a sequence, so the array is wrapped in a tuple by hand
        (position,) = positions

        def score_from_one_array(contributions: Sequence[NDArray[np.float32]]) -> float:
            return compute_objective_score((contributions[position],))

        return score_from_one_array
    else:
        pick_objective_arrays = operator.itemgetter(*positions)

        def score_from_picked_arrays(contributions: Sequence[NDArray[np.float32]]) -> float:
            return compute_objective_score(pick_objective_arrays(contributions))

        return score_from_picked_arrays


def _separating_tie_breaker_metrics(diversity_metric: DiversityMetric) -> tuple[DiversityMetric, ...]:
    """Return the diversity metrics to use as tie-breakers when the caller sets none, by the main diversity metric."""
    if diversity_metric == DiversityMetric.MIN_SEPARATION:
        # min-separation reacts only to the closest pair; the approximate geomean rewards a uniform
        # spread, which opens room around that pair so the minimum separation itself can grow.
        return (DiversityMetric.APPROX_GEOMEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC)
    if diversity_metric in (DiversityMetric.GEOMEAN_SEPARATION, DiversityMetric.APPROX_GEOMEAN_SEPARATION):
        # once more than one pair coincides the geomean is stuck at zero; the non-zero fraction
        # rewards cutting the count of coincident pairs, a path back toward a non-zero geomean.
        return (DiversityMetric.NON_ZERO_SEPARATION_FRAC,)
    return ()
