"""A solver's configuration is held apart from the distances it will read.

The distance store and the rest of a solver are separated because the distance store is built once
and read by several processes, while each process assembles its own solver over it from a copy of
this record — which is why the record must stay small enough to pickle.
"""

from dataclasses import dataclass, replace

from max_div._core.constraints import Constraint
from max_div._core.metrics import DiversityMetric
from max_div._core.metrics._distance import DistanceStore

from ._constraint_penalty import ConstraintPenalty
from ._solver import MaxDivSolver
from ._solver_step import SolverStep


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

    def build_solver(self, store: DistanceStore) -> MaxDivSolver:
        """Return a solver configured as this record describes, reading the given store."""
        return MaxDivSolver(
            n=self.n,
            store=store,
            k=self.k,
            diversity_metric=self.diversity_metric,
            diversity_tie_breakers=self.diversity_tie_breakers,
            constraints=self.constraints,
            solver_steps=self.solver_steps,
            seed=self.seed,
            constraint_penalty=self.constraint_penalty,
            distance_storage_label=self.distance_storage_label,
        )

    def with_seed(self, seed: int) -> "SolverConfig":
        """Return a copy of this configuration carrying a different seed."""
        return replace(self, seed=seed)
