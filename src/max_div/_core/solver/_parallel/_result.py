"""A worker sends back a result, and `best_result` picks the winner among several."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score


@dataclass(frozen=True)
class WorkerResult:
    """A result records one worker's selection, what it scored, and what it cost."""

    worker_index: int
    i_selected: NDArray[np.int32]
    score: Score
    elapsed: Elapsed
    seed: int


def best_result(results: list[WorkerResult]) -> WorkerResult:
    """Return the highest-scoring result, ties going to the lowest worker index.

    Ties break by index rather than by arrival, so the same seeds give the same winner every run.

    :param results: what each worker reported; workers that failed are simply absent.
    :raises ValueError: If no results were collected, which means every worker failed.
    """
    if not results:
        raise ValueError("A portfolio returned no results at all; every worker failed to report one.")
    return max(results, key=lambda result: (result.score, -result.worker_index))
