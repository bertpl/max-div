"""A solver's configuration is held apart from the distances it will read.

The distance store and the rest of a solver are separated because the distance store is built once
and read by several processes, while each process assembles its own solver over it from a copy of
this record — which is why the record must stay small enough to pickle.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from max_div._core.constraints import Constraint
from max_div._core.metrics import DiversityObjective
from max_div._core.metrics._distance import DistanceStore

from ._constraint_penalty import ConstraintPenalty
from ._distance_storage import DistanceStorageTypes, StoreDistance
from ._duration import E2eBudget
from ._solver import MaxDivSolver
from ._solver_step import REPORTING_BATCH_SECONDS, SolverStep


@dataclass(frozen=True)
class SolverConfig:
    """A config holds everything a solver needs apart from the distances it reads."""

    n: int
    k: int
    # the primary objective first, then the tie-breakers in order
    diversity_objectives: list[DiversityObjective]
    constraints: list[Constraint]
    solver_steps: list[SolverStep]
    seed: int
    constraint_penalty: ConstraintPenalty
    distance_storage: DistanceStorageTypes
    # `batch_seconds` targets the wall-clock size of one optimization batch (set per worker by
    # the parallel builder)
    batch_seconds: float = REPORTING_BATCH_SECONDS
    # an end-to-end budget bounds the whole solve, distance computation and initialization
    # included; the parallel solver replaces it with a started copy at its own solve start, so
    # workers charge the parent's setup against the budget too
    e2e_budget: E2eBudget | None = None

    def build_solver(
        self,
        *,
        stores_by_distance: Mapping[StoreDistance, DistanceStore] | None = None,
        stores_by_distance_provider: Callable[[], Mapping[StoreDistance, DistanceStore]] | None = None,
    ) -> MaxDivSolver:
        """Return a solver configured as this record describes, given the distances it will read.

        Pass exactly one of:

        Args:
            stores_by_distance: an already-built distance -> store mapping — the parallel solver's
                workers attach to the shared stores and hand the mapping in.
            stores_by_distance_provider: a callable that yields the mapping when the solve starts,
                so `build` stays lean and the stores are built inside `solve`.

        Raises:
            ValueError: if neither or both are given, or a step's strategy does not support the
                diversity metric.
        """
        for step in self.solver_steps:
            step.validate_objective(self.diversity_objectives[0])
        if stores_by_distance is not None and stores_by_distance_provider is None:
            provider: Callable[[], Mapping[StoreDistance, DistanceStore]] = lambda: stores_by_distance
        elif stores_by_distance is None and stores_by_distance_provider is not None:
            provider = stores_by_distance_provider
        else:
            raise ValueError("Pass exactly one of `stores_by_distance` or `stores_by_distance_provider`.")
        return MaxDivSolver(
            n=self.n,
            stores_by_distance_provider=provider,
            k=self.k,
            diversity_objectives=self.diversity_objectives,
            constraints=self.constraints,
            solver_steps=self.solver_steps,
            seed=self.seed,
            constraint_penalty=self.constraint_penalty,
            distance_storage=self.distance_storage,
            batch_seconds=self.batch_seconds,
            e2e_budget=self.e2e_budget,
        )

    def with_seed(self, seed: int) -> "SolverConfig":
        """Return a copy of this configuration carrying a different seed."""
        return replace(self, seed=seed)

    def with_e2e_budget(self, e2e_budget: E2eBudget) -> "SolverConfig":
        """Return a copy of this configuration carrying the given end-to-end budget."""
        return replace(self, e2e_budget=e2e_budget)
