"""A worker sends back a result or a failure, and `best_result` picks the winner among the results.

A worker reports once, when it finishes, so the result carries its whole solution rather than a
trimmed selection and score.  A worker whose solve raises reports a `WorkerFailure` instead, so
the parent can name which worker failed and why.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score
from max_div._core.solver._solution import MaxDivSolution

from ._worker_group_change import WorkerGroupChange


@dataclass(frozen=True)
class WorkerFailure:
    """A failure records which worker raised, and the exception with its full traceback text."""

    worker_index: int
    error: str
    traceback_text: str


@dataclass(frozen=True)
class WorkerResult:
    """A result records which worker ran, with which seed, when it started, what it reached, and what it regrouped.

    `t_start` is the `time.monotonic()` reading the worker took just before its solve started; the
    parent uses it to place this worker's checkpoints and worker group changes next to the other
    workers' (see `_trajectory` and `_group_history`). `worker_group_changes` are the dissolutions
    this worker executed, with `elapsed` counted from the worker's own start.
    """

    worker_index: int
    seed: int
    t_start: float
    solution: MaxDivSolution
    worker_group_changes: list[WorkerGroupChange] = field(default_factory=list)

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


def earliest_start_time(results: list[WorkerResult]) -> float:
    """Return the earliest worker start time: the zero of the axis that a parallel solution's checkpoints share."""
    return min(result.t_start for result in results)


def best_result(results: list[WorkerResult], failures: list[WorkerFailure] | None = None) -> WorkerResult:
    """Return the highest-scoring result, ties going to the lowest worker index.

    The same seeds give the same winner every run, whichever worker reports first.

    Args:
        results: what each worker reported; workers that failed are simply absent.
        failures: what failed workers reported, if any; when every worker failed, the first
            failure's traceback is included in the raised error so the cause is visible.

    Raises:
        ValueError: If no results were collected, which means every worker failed.
    """
    if not results:
        detail = ""
        if failures:
            detail = f"\n\nWorker {failures[0].worker_index}'s failure:\n{failures[0].traceback_text}"
        raise ValueError(f"A parallel solve returned no results at all; every worker failed to report one.{detail}")
    return max(results, key=lambda result: (result.score, -result.worker_index))
