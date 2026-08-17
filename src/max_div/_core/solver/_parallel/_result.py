"""A worker sends back a result, and `best_result` picks the winner among several.

A worker reports once, when it finishes, so the result carries its whole solution rather than a
trimmed selection and score.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score
from max_div._core.solver._solution import MaxDivSolution


@dataclass(frozen=True)
class WorkerResult:
    """A result records which worker ran, with which seed, and the solution it reached."""

    worker_index: int
    seed: int
    solution: MaxDivSolution

    @property
    def score(self) -> Score:
        """Return the score of the solution this worker reached."""
        return self.solution.score

    @property
    def i_selected(self) -> NDArray[np.int32]:
        """Return the items this worker selected."""
        return self.solution.i_selected

    @property
    def elapsed(self) -> Elapsed:
        """Return the time and iterations this worker spent."""
        return self.solution.duration


def best_result(results: list[WorkerResult]) -> WorkerResult:
    """Return the highest-scoring result, ties going to the lowest worker index.

    The same seeds give the same winner every run, whichever worker reports first.

    Args:
        results: what each worker reported; workers that failed are simply absent.

    Raises:
        ValueError: If no results were collected, which means every worker failed.
    """
    if not results:
        raise ValueError("A portfolio returned no results at all; every worker failed to report one.")
    return max(results, key=lambda result: (result.score, -result.worker_index))
