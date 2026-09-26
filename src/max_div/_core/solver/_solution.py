from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ._distance_storage import DistanceStorageTypes
from ._duration import Elapsed
from ._score import Score
from ._score_checkpoint import ScoreCheckpoint


@dataclass
class MaxDivSolution:
    """Result of solving a Maximum Diversity Problem.

    Contains the selected item indices, the final [`Score`][max_div.solution.Score], timing information,
    and a history of score checkpoints recorded during the solve.
    """

    # --- final solution -------------------------
    i_selected: NDArray[np.int32]

    # --- score & checkpoints --------------------
    # The checkpoints are in solve order; the last one is the final state.
    score_checkpoints: list[ScoreCheckpoint]

    @property
    def score(self) -> Score:
        """Return the final score of the solution."""
        return self.score_checkpoints[-1].score

    # --- durations ------------------------------
    # one per solver step, in step order; index 0 is the solver state initialization
    step_durations: list[Elapsed]

    @property
    def duration(self) -> Elapsed:
        """Return the total elapsed time and iterations taken to compute the solution.

        The time includes the solver's own work before and between steps, so it can exceed the
        sum of `step_durations`; for a single solve, the iteration count equals their sum.
        """
        return self.score_checkpoints[-1].elapsed

    # --- diversity objectives -------------------
    # Each diversity objective gets one label, the primary objective first, then the tie-breakers in
    # the order they break ties; the list runs parallel to every score's `diversities`.
    diversity_objective_labels: list[str] = field(default_factory=list)

    # --- constraints ----------------------------
    n_constraints: int = 0
    n_constraints_satisfied: int = 0

    # --- distance storage -----------------------
    # how each distance store was stored; empty when unreported
    distance_storage: DistanceStorageTypes = field(default_factory=DistanceStorageTypes)

    # --- string representation ------------------
    def __str__(self) -> str:
        parts = [
            f"MaxDivSolution: {len(self.i_selected)} items selected",
            f"diversity={self.score.diversity:.4f}",
        ]
        if self.n_constraints > 0:
            parts.append(f"constraints: {self.n_constraints_satisfied}/{self.n_constraints} satisfied")
        if self.distance_storage.per_store:
            parts.append(f"storage={self.distance_storage}")
        parts.append(str(self.duration))
        return " | ".join(parts)
