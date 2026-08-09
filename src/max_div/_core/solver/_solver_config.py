"""A solver's configuration, held apart from the distances it will read.

Everything `MaxDivSolver` needs except its store, in one record small enough to pickle.  Building a
solver is two steps for that reason: the distances are produced once and can be shared between
processes, while each process assembles its own solver over them from a copy of this record.
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
    """Everything a solver needs apart from the distances it reads.

    Holds no distances and no problem, so it stays small however large the problem is — which is
    what lets it travel to a worker process while the distances travel through shared memory.
    """

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
