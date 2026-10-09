"""A solver's configuration is held apart from the data matrices that its distance stores read.

The data matrices and the rest of a solver are separated because the data matrices are produced once
and read by several processes, while each process assembles its own solver over them from a copy of
this record — which is why the record must stay small enough to pickle.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from max_div._core.constraints import Constraint
from max_div._core.distance_storage import DistanceStorageTypes
from max_div._core.metrics import DiversityObjective
from max_div._core.metrics._distance import DataMatrixReader

from ._constraint_penalty import ConstraintPenalty
from ._duration import E2eBudget
from ._solver import MaxDivSolver
from ._solver_step import REPORTING_BATCH_SECONDS, SolverStep


@dataclass(frozen=True)
class SolverConfig:
    """A config holds everything that a solver needs apart from the data matrices that its distance stores read."""

    n: int
    k: int
    # The primary objective comes first, then the tie-breakers in order; each distance spec names the data
    # matrix that its distance store reads.
    diversity_objectives: list[DiversityObjective]
    constraints: Sequence[Constraint]
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
        data_matrix_reader: DataMatrixReader | None = None,
        data_matrix_reader_provider: Callable[[], DataMatrixReader] | None = None,
    ) -> MaxDivSolver:
        """Return a solver configured as this record describes, given the reader of its data matrices.

        Pass exactly one of:

        Args:
            data_matrix_reader: a reader over data matrices that exist already — a worker of the
                parallel solver reads the ones that its parent published in shared memory.
            data_matrix_reader_provider: a callable that returns the reader when the solve starts,
                so `build` stays cheap and the data matrices are produced inside `solve`.

        Raises:
            ValueError: if neither or both are given.
        """
        if data_matrix_reader is not None and data_matrix_reader_provider is None:
            provider: Callable[[], DataMatrixReader] = lambda: data_matrix_reader
        elif data_matrix_reader is None and data_matrix_reader_provider is not None:
            provider = data_matrix_reader_provider
        else:
            raise ValueError("Pass exactly one of `data_matrix_reader` or `data_matrix_reader_provider`.")
        return MaxDivSolver(
            n=self.n,
            data_matrix_reader_provider=provider,
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
