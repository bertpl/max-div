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

    def build_solver(self, store: DistanceStore) -> MaxDivSolver:
        """Return a solver configured as this record describes, reading an already-built store.

        The portfolio uses this: its workers attach to the shared store and hand it in.
        """
        return self.build_solver_deferred(lambda: store)

    def build_solver_deferred(self, store_provider: Callable[[], DistanceStore]) -> MaxDivSolver:
        """Return a solver that obtains its store from `store_provider` when it solves.

        A single solve builds its store this way, so the build is part of the solve rather than
        done up front.
        """
        return MaxDivSolver(
            n=self.n,
            store_provider=store_provider,
            k=self.k,
            diversity_metric=self.diversity_metric,
            diversity_tie_breakers=self.diversity_tie_breakers,
            constraints=self.constraints,
            solver_steps=self.solver_steps,
            seed=self.seed,
            constraint_penalty=self.constraint_penalty,
            distance_storage_label=self.distance_storage_label,
            batch_seconds=self.batch_seconds,
        )

    def with_seed(self, seed: int) -> "SolverConfig":
        """Return a copy of this configuration carrying a different seed."""
        return replace(self, seed=seed)
