"""What a portfolio returns: the winning selection, and what every worker found.

The winner is an ordinary solution — the same type a single solve returns, so code that accepts one
keeps working.  What a portfolio adds is the per-worker record, which is the only way a user can
tell whether running several workers bought anything.
"""

from dataclasses import dataclass, field

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score
from max_div._core.solver._solution import MaxDivSolution

from ._worker_config import WorkerConfig


@dataclass(frozen=True)
class WorkerSummary:
    """What one worker ran, what it found, and what it cost.

    Carries the configuration by value rather than by reference, so a saved result can be reproduced
    without the code that produced it: the seed alone does not say which solver to replay it with.
    """

    worker_index: int
    config: WorkerConfig
    seed: int
    score: Score
    elapsed: Elapsed
    has_best_score: bool


@dataclass
class ParallelMaxDivSolution(MaxDivSolution):
    """The winning worker's solution, plus a summary of what every worker found."""

    workers: list[WorkerSummary] = field(default_factory=list)
    winning_worker: int = 0

    @property
    def n_workers_with_best_score(self) -> int:
        """Return how many workers reached the best score, the winner included.

        Equal to the worker count when every worker tied, which is the signal that the portfolio
        bought nothing on this problem.
        """
        return sum(1 for worker in self.workers if worker.has_best_score)

    def __str__(self) -> str:
        """Return the single-solve summary, plus how many workers reached the best score.

        That count is the one number saying whether running several workers bought anything: equal
        to the worker count means they all found equally good selections.
        """
        return (
            f"{super().__str__()}, best score reached by {self.n_workers_with_best_score}/{len(self.workers)} workers"
        )
