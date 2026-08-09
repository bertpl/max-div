"""A portfolio returns the winning selection together with what every worker found."""

from dataclasses import dataclass, field

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score
from max_div._core.solver._solution import MaxDivSolution

from ._worker_config import WorkerConfig


@dataclass(frozen=True)
class WorkerSummary:
    """A summary records what one worker ran, what it found, and what it cost.

    The configuration is carried by value rather than by reference, so a saved result can be replayed
    without the code that produced it: a seed alone does not say which solver to replay it with.
    """

    worker_index: int
    config: WorkerConfig
    seed: int
    score: Score
    elapsed: Elapsed
    has_best_score: bool


@dataclass
class ParallelMaxDivSolution(MaxDivSolution):
    """A parallel solution is the winning worker's, with a summary of every worker attached.

    Being an ordinary `MaxDivSolution` is deliberate: code written for a single solve keeps working.
    """

    workers: list[WorkerSummary] = field(default_factory=list)
    winning_worker: int = 0

    @property
    def n_workers_with_best_score(self) -> int:
        """Return how many workers reached the best score, the winner included.

        Equal to the worker count when every worker tied, which means the portfolio found nothing a
        single worker would not have.
        """
        return sum(1 for worker in self.workers if worker.has_best_score)

    def __str__(self) -> str:
        """Return the single-solve summary, plus how many workers reached the best score."""
        return (
            f"{super().__str__()} | best score reached by {self.n_workers_with_best_score}/{len(self.workers)} workers"
        )
