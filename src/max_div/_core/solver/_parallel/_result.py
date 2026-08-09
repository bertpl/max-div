"""What a worker sends back, and how the best of several is chosen.

The payload is the selection and its score rather than a whole solution: a mode that shares an
incumbent mid-run sends this same record far more often, so it is kept to what a receiver can act
on.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score


@dataclass(frozen=True)
class WorkerResult:
    """One worker's selection, what it scored, and what it cost to get there."""

    worker_index: int
    i_selected: NDArray[np.int32]
    score: Score
    elapsed: Elapsed
    seed: int


def best_result(results: list[WorkerResult]) -> WorkerResult:
    """Return the winning result: the highest score, and among equal scores the lowest worker index.

    Breaking ties by index rather than by arrival makes the winner a property of the results instead
    of a race between processes, so a portfolio run twice over the same seeds returns the same
    selection both times.

    Raises:
        ValueError: If no results were collected, which means every worker failed.
    """
    if not results:
        raise ValueError("A portfolio returned no results at all; every worker failed to report one.")
    return max(results, key=lambda result: (result.score, -result.worker_index))
