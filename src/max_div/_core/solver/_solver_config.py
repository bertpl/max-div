"""A solver's configuration is held apart from the distances it will read.

The distance store and the rest of a solver are separated because the distance store is built once
and read by several processes, while each process assembles its own solver over it from a copy of
this record — which is why the record must stay small enough to pickle.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from max_div._core.constraints import Constraint
from max_div._core.metrics import DiversityMetric
from max_div._core.metrics._distance import DistanceStore

from ._constraint_penalty import ConstraintPenalty
from ._solver import MaxDivSolver
from ._solver_step import REPORTING_BATCH_SECONDS, SolverStep


@dataclass(frozen=True)
class SolverConfig:
    """A config holds everything a solver needs apart from the distances it reads."""

    n: int
    k: int
    diversity_metric: DiversityMetric
    diversity_tie_breakers: list[DiversityMetric]
    constraints: list[Constraint]
    solver_steps: list[SolverStep]
    seed: int
    constraint_penalty: ConstraintPenalty
    distance_storage_label: str
    # `batch_seconds` targets the wall-clock size of one optimization batch (set per worker by
    # the parallel builder)
    batch_seconds: float = REPORTING_BATCH_SECONDS
    # an end-to-end budget bounds the whole solve (store build and initialization included) with
    # `e2e_budget_sec`; `t_e2e_budget_start` is the `time.monotonic()` reading the parallel solver
    # stamps at its own solve start, so workers charge the parent's setup against the budget too
    e2e_budget_sec: float | None = None
    t_e2e_budget_start: float | None = None

    def build_solver(
        self,
        *,
        store: DistanceStore | None = None,
        store_provider: Callable[[], DistanceStore] | None = None,
    ) -> MaxDivSolver:
        """Return a solver configured as this record describes, given the distances it will read.

        Pass exactly one of:

        Args:
            store: an already-built store to read from — the parallel solver's workers attach to
                the shared store and hand it in.
            store_provider: a callable that yields the store when the solve starts, so `build`
                stays lean and the store is built inside `solve`.

        Raises:
            ValueError: if neither or both are given.
        """
        if store is not None and store_provider is None:
            provider: Callable[[], DistanceStore] = lambda: store
        elif store is None and store_provider is not None:
            provider = store_provider
        else:
            raise ValueError("Pass exactly one of `store` or `store_provider`.")
        return MaxDivSolver(
            n=self.n,
            store_provider=provider,
            k=self.k,
            diversity_metric=self.diversity_metric,
            diversity_tie_breakers=self.diversity_tie_breakers,
            constraints=self.constraints,
            solver_steps=self.solver_steps,
            seed=self.seed,
            constraint_penalty=self.constraint_penalty,
            distance_storage_label=self.distance_storage_label,
            batch_seconds=self.batch_seconds,
            e2e_budget_sec=self.e2e_budget_sec,
            t_e2e_budget_start=self.t_e2e_budget_start,
        )

    def with_seed(self, seed: int) -> "SolverConfig":
        """Return a copy of this configuration carrying a different seed."""
        return replace(self, seed=seed)

    def with_t_e2e_budget_start(self, t_e2e_budget_start: float) -> "SolverConfig":
        """Return a copy of this configuration carrying the given budget start time (`time.monotonic()`)."""
        return replace(self, t_e2e_budget_start=t_e2e_budget_start)
