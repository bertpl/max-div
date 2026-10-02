"""A solver's configuration is held apart from the distances it will read.

The distance store and the rest of a solver are separated because the distance store is built once
and read by several processes, while each process assembles its own solver over it from a copy of
this record — which is why the record must stay small enough to pickle.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from max_div._core.constraints import Constraint
from max_div._core.distance_storage import DistanceStorageTypes
from max_div._core.metrics import DistanceMetric, DiversityObjective
from max_div._core.metrics._distance import DistanceStore

from ._constraint_penalty import ConstraintPenalty
from ._diversity_contribution import DiversityObjectiveBindings
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
    # When on, every score checkpoint also carries the selection held at that moment.
    intermediate_selections_enabled: bool = False

    def build_solver(
        self,
        *,
        stores: Sequence[DistanceStore] | None = None,
        stores_provider: Callable[[], Sequence[DistanceStore]] | None = None,
    ) -> MaxDivSolver:
        """Return a solver configured as this record describes, given its distance stores.

        The stores come in the store order of `DiversityObjectiveBindings` over this config's
        `diversity_objectives`; this method pairs each one with its distance metric, so a worker that
        attaches the published stores rebuilds the same mapping as the process that built them.

        Pass exactly one of:

        Args:
            stores: already-built distance stores — the parallel solver's workers attach to the
                shared stores and hand them in.
            stores_provider: a callable that returns the stores when the solve starts, so that the
                builder's `build` returns quickly and the stores are built inside `solve`.

        Raises:
            ValueError: if neither or both are given.
        """
        bindings = DiversityObjectiveBindings.for_objectives(self.diversity_objectives)
        if stores is not None and stores_provider is None:
            stores_by_distance = bindings.stores_by_distance(stores)
            provider: Callable[[], Mapping[DistanceMetric | None, DistanceStore]] = lambda: stores_by_distance
        elif stores is None and stores_provider is not None:
            provider = lambda: bindings.stores_by_distance(stores_provider())
        else:
            raise ValueError("Pass exactly one of `stores` or `stores_provider`.")
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
            intermediate_selections_enabled=self.intermediate_selections_enabled,
        )

    def with_seed(self, seed: int) -> "SolverConfig":
        """Return a copy of this configuration carrying a different seed."""
        return replace(self, seed=seed)

    def with_e2e_budget(self, e2e_budget: E2eBudget) -> "SolverConfig":
        """Return a copy of this configuration carrying the given end-to-end budget."""
        return replace(self, e2e_budget=e2e_budget)
