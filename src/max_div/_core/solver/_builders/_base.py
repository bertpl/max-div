"""A solver builder and a portfolio builder both carry the settings defined here.

Settings are split by one rule: **anything that influences the score is shared**.  A portfolio
compares what its workers found, so one answer to "which selection is better" has to hold across all
of them.

Distance storage is here for a different reason: workers read one buffer, so per-worker storage
could not be honored at all.

A subclass adds the search: which strategies run, and for how long.
"""

from typing import TYPE_CHECKING, Self

from max_div._core.metrics import DiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._constraint_penalty import ConstraintPenalty
from max_div._core.solver._distance_storage import (
    DistanceStorage,
    select_distance_storage,
    total_physical_memory_bytes,
)

if TYPE_CHECKING:
    from max_div._core.constraints import Constraint


class SolverBuilderBase:
    """A builder base holds the settings that define the score, plus the problem facts every solver needs."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, problem: MaxDivProblem) -> None:
        """Configure a builder over the given problem."""
        # --- problem ---------------------------
        self._problem = problem

        # --- problem properties ----------------
        self._n: int = problem.n
        self._k: int = problem.k
        self._diversity_metric: DiversityMetric = problem.diversity_metric
        self._constraints: list[Constraint] = problem.constraints

        # --- shared configuration --------------
        self._diversity_tie_breakers: list[DiversityMetric] = []
        self._default_diversity_tie_breakers: bool = True
        self._seed = 42
        self._constraint_penalty: ConstraintPenalty = ConstraintPenalty.LINEAR
        self._distance_storage: DistanceStorage = DistanceStorage.AUTO

    # -------------------------------------------------------------------------
    #  Shared builder API
    # -------------------------------------------------------------------------
    def with_diversity_tie_breakers(self, diversity_tie_breakers: list[DiversityMetric]) -> Self:
        """Set custom diversity tie-breaker metrics, overriding the defaults."""
        self._diversity_tie_breakers = diversity_tie_breakers
        self._default_diversity_tie_breakers = False
        return self

    def with_default_diversity_tie_breakers(self) -> Self:
        """Reset to automatically chosen tie-breakers based on the main diversity metric."""
        self._diversity_tie_breakers = []
        self._default_diversity_tie_breakers = True
        return self

    def with_seed(self, seed: int) -> Self:
        """Set the random seed for reproducibility (default: 42)."""
        self._seed = seed
        return self

    def with_constraint_penalty(self, penalty: ConstraintPenalty) -> Self:
        """Set how constraint violations are penalized in the feasibility score (default: LINEAR)."""
        self._constraint_penalty = penalty
        return self

    def with_distance_storage(self, storage: DistanceStorage) -> Self:
        """Set how pairwise distances are stored during search (default: DistanceStorage.AUTO)."""
        self._distance_storage = storage
        return self

    # -------------------------------------------------------------------------
    #  Resolution
    # -------------------------------------------------------------------------
    def _select_storage(self) -> tuple[DistanceStorage, str]:
        """Return the backend this configuration selects, and the label reported to the user."""
        resolved = select_distance_storage(self._problem, self._distance_storage, total_physical_memory_bytes())
        return resolved, resolved.value + (" (auto)" if self._distance_storage == DistanceStorage.AUTO else "")

    def _determine_diversity_tie_breakers(self) -> list[DiversityMetric]:
        """Return the tie-breakers to score with: the user's if set, otherwise per the main metric."""
        if not self._default_diversity_tie_breakers:
            return self._diversity_tie_breakers
        if self._diversity_metric == DiversityMetric.MIN_SEPARATION:
            return [
                DiversityMetric.APPROX_GEOMEAN_SEPARATION,
                DiversityMetric.NON_ZERO_SEPARATION_FRAC,
            ]
        if (self._diversity_metric == DiversityMetric.GEOMEAN_SEPARATION) or (
            self._diversity_metric == DiversityMetric.APPROX_GEOMEAN_SEPARATION
        ):
            return [DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        return []
