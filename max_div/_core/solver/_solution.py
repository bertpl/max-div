from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score


@dataclass
class MaxDivSolution:
    """
    Result of solving a Maximum Diversity Problem.

    Contains the selected vector indices, the final [`Score`][max_div.solution.Score], timing information,
    and a history of score checkpoints recorded during the solve.
    """

    # --- final solution ----------------------------------
    i_selected: NDArray[np.int32]

    # --- score & checkpoints -----------------------------
    # list of (step_name, elapsed, score) tuples
    # where elapsed times/iterations are cumulative metrics starting at the start of the first solver step
    score_checkpoints: list[tuple[str, Elapsed, Score]]

    @property
    def score(self) -> Score:
        """Return the final score of the solution."""
        return self.score_checkpoints[-1][2]

    # --- durations ---------------------------------------
    step_durations: dict[str, Elapsed]

    @property
    def duration(self) -> Elapsed:
        """Return the total elapsed time and iterations taken to compute the solution."""
        return self.score_checkpoints[-1][1]

    # --- constraints -------------------------------------
    n_constraints: int = 0
    n_constraints_satisfied: int = 0

    # --- string representation ---------------------------
    def __str__(self) -> str:
        parts = [
            f"MaxDivSolution: {len(self.i_selected)} vectors selected",
            f"diversity={self.score.diversity:.4f}",
        ]
        if self.n_constraints > 0:
            parts.append(f"constraints: {self.n_constraints_satisfied}/{self.n_constraints} satisfied")
        parts.append(str(self.duration))
        return " | ".join(parts)
