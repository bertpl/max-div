"""A parallel solve returns the winning selection together with what every worker found."""

from dataclasses import dataclass, field

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score
from max_div._core.solver._solution import MaxDivSolution

from ._worker_config import WorkerConfig
from ._worker_group_change import WorkerGroupChange


@dataclass(frozen=True)
class WorkerSummary:
    """A summary records what one worker ran, what it found, when it started, and what it cost.

    The summary stores the worker's whole configuration, not just its seed, so a saved result can be
    replayed without the code that produced it: a seed alone does not say which solver to replay with.

    `t_start_offset_sec` is when the worker started on the parallel solve's shared time axis: seconds
    since the earliest worker started, and zero for that earliest worker. The solution's
    `score_checkpoints` and `worker_group_changes` are placed on that same axis.
    """

    worker_index: int
    config: WorkerConfig
    seed: int
    score: Score
    elapsed: Elapsed
    has_best_score: bool
    t_start_offset_sec: float = 0.0


@dataclass
class ParallelMaxDivSolution(MaxDivSolution):
    """A parallel solution is the winning worker's selection, with a summary of every worker attached.

    Its `score_checkpoints` do not trace the winning worker's own solve: they trace the best score any
    worker held at each moment, each checkpoint naming that worker and its group, on one time axis
    whose zero is the earliest worker start. `duration` therefore spans the whole parallel solve,
    while its iteration count is that of the worker holding the last checkpoint. `step_durations`
    stay the winning worker's.

    `initial_worker_groups` gives each worker's group at the start, in worker order, and
    `worker_group_changes` every dissolution since, in the order they happened and on the same time
    axis as the checkpoints; replaying the changes over the initial groups gives the grouping at any
    moment. A fixed grouping has no changes.

    Subclassing `MaxDivSolution` keeps code written for a single solve working.
    """

    workers: list[WorkerSummary] = field(default_factory=list)
    winning_worker: int = 0
    initial_worker_groups: list[int] = field(default_factory=list)
    worker_group_changes: list[WorkerGroupChange] = field(default_factory=list)

    @property
    def n_workers_with_best_score(self) -> int:
        """Return how many workers reached the best score, the winner included.

        The count equals the worker count when every worker tied, which means the parallel solve found
        nothing a single worker would not have.
        """
        return sum(1 for worker in self.workers if worker.has_best_score)

    def __str__(self) -> str:
        """Return the single-solve summary, plus how many workers reached the best score."""
        return (
            f"{super().__str__()} | best score reached by {self.n_workers_with_best_score}/{len(self.workers)} workers"
        )
