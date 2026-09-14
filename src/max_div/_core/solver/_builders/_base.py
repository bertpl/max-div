"""A single-solver builder and a parallel-solver builder both carry the settings defined here.

Settings are split by one rule: **anything that influences the score is shared**.  A parallel solver
compares what its workers found, so one answer to "which selection is better" has to hold across all
of them.

Distance storage is here for a different reason: workers read one buffer, so per-worker storage
could not be honored at all.

A subclass adds the search: which strategies run, and for how long.
"""

from typing import TYPE_CHECKING, Self

from max_div._core.metrics import (
    DiversityMetric,
    DiversityObjective,
    DiversityObjectiveSimple,
)
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._constraint_penalty import ConstraintPenalty
from max_div._core.solver._distance_storage import (
    DistanceStorageType,
    DistanceStorageTypes,
    DistanceStoreFactory,
    total_physical_memory_bytes,
)
from max_div._core.solver._diversity_contribution import DiversityObjectiveBindings
from max_div._core.solver._duration import E2eBudget, TargetDuration, TargetTimeDuration

if TYPE_CHECKING:
    from max_div._core.constraints import Constraint


class SolverBuilderBase:
    """A builder base holds the settings that define the score, plus the problem facts every solver needs."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, problem: MaxDivProblem) -> None:
        """Configure a builder over the given problem."""
        # --- problem ----------------------------
        self._problem = problem

        # --- problem properties -----------------
        self._n: int = problem.n
        self._k: int = problem.k
        self._primary_objective: DiversityObjective = problem.diversity_objective
        self._constraints: list[Constraint] = problem.constraints

        # --- shared configuration ---------------
        self._custom_diversity_tie_breakers: list[DiversityObjective] | None = None  # None → the defaults
        self._seed = 42
        self._constraint_penalty: ConstraintPenalty = ConstraintPenalty.LINEAR
        self._distance_storage_type: DistanceStorageType = DistanceStorageType.AUTO
        self._e2e_enabled: bool = False
        self._target_duration: TargetDuration | None = None

    # -------------------------------------------------------------------------
    #  Shared builder API
    # -------------------------------------------------------------------------
    def with_diversity_tie_breakers(self, diversity_tie_breaker_metrics: list[DiversityMetric]) -> Self:
        """Set custom diversity tie-breaker metrics, overriding the defaults.

        Each tie-breaker reads the problem's own distance, which a hybrid diversity metric has several
        of, so a problem over a hybrid keeps its default tie-breakers.

        Raises:
            ValueError: If the problem's diversity metric is a hybrid.
        """
        if not isinstance(self._primary_objective, DiversityObjectiveSimple):
            raise ValueError("Custom diversity tie-breakers are not supported for a hybrid diversity metric.")
        self._custom_diversity_tie_breakers = [
            DiversityObjectiveSimple(metric, self._primary_objective.distance_metric)
            for metric in diversity_tie_breaker_metrics
        ]
        return self

    def with_default_diversity_tie_breakers(self) -> Self:
        """Reset to automatically chosen tie-breakers based on the primary diversity metric."""
        self._custom_diversity_tie_breakers = None
        return self

    def with_seed(self, seed: int) -> Self:
        """Set the random seed for reproducibility (default: 42)."""
        self._seed = seed
        return self

    def with_constraint_penalty(self, penalty: ConstraintPenalty) -> Self:
        """Set how constraint violations are penalized in the feasibility score (default: LINEAR)."""
        self._constraint_penalty = penalty
        return self

    def with_distance_storage(self, storage_type: DistanceStorageType) -> Self:
        """Set how pairwise distances are stored during search (default: DistanceStorageType.AUTO)."""
        self._distance_storage_type = storage_type
        return self

    def with_end_to_end_budget(self, enabled: bool = True) -> Self:
        """Make the configured budget bound the whole solve, setup included (default: per-step budgets).

        With the end-to-end budget enabled, the budget also covers distance computation,
        initialization, and — for a parallel solve — worker spawning, and the optimization gets
        whatever time remains.
        The budget itself is the one configured on `with_preset` / `with_workers` /
        `with_custom_worker_groups`; `build()` rejects the combination with an iteration budget.
        """
        self._e2e_enabled = enabled
        return self

    # -------------------------------------------------------------------------
    #  Resolution
    # -------------------------------------------------------------------------
    def _resolve_e2e_budget(self) -> E2eBudget | None:
        """Return the end-to-end budget this configuration asks for, or None; called at build time.

        Raises:
            ValueError: If the end-to-end budget is enabled without a time budget — not all
                solver phases can be expressed in iteration counts.
        """
        if not self._e2e_enabled:
            return None
        if not isinstance(self._target_duration, TargetTimeDuration):
            raise ValueError(
                "with_end_to_end_budget requires a time budget (seconds/minutes/hours); "
                "not all solver phases can be expressed in iteration counts."
            )
        return E2eBudget(budget_sec=self._target_duration.value())

    def _store_factory(self) -> tuple[DistanceStoreFactory, DistanceStorageTypes]:
        """Return the store factory and each store's resolved (distance, storage type)."""
        bindings = DiversityObjectiveBindings.for_objectives(self._determine_diversity_objectives())
        factory = DistanceStoreFactory(
            self._problem,
            bindings.distance_metrics,
            self._distance_storage_type,
            total_physical_memory_bytes(),
        )
        return factory, factory.resolved_storage()

    def _determine_diversity_objectives(self) -> list[DiversityObjective]:
        """Return the diversity objectives, the primary objective first and then its tie-breakers."""
        return [self._primary_objective, *self._determine_diversity_tie_breakers()]

    def _determine_diversity_tie_breakers(self) -> list[DiversityObjective]:
        """Return the tie-breaker objectives to rank ties by: the user's, or the primary objective's defaults."""
        if self._custom_diversity_tie_breakers is not None:
            return self._custom_diversity_tie_breakers
        else:
            return self._primary_objective.default_tie_breakers()
