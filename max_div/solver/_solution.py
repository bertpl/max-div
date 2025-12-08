from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from max_div.solver._duration import Elapsed
from max_div.solver._score import Score


@dataclass
class MaxDivSolution:
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
