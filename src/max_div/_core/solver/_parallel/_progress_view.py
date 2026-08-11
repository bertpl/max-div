"""The parent side of the progress channel: one composite view over every worker's snapshots.

A rendered row never describes a single worker — it combines two independently sourced halves:

- **Progress** follows the slowest still-running worker. That is the progress of the call itself
  (`solve` returns when the last worker finishes), so the fraction answers "how much longer" and is
  monotone by construction. Workers known to have died are excluded, so the view cannot freeze.
- **Result** shows the best score reached so far by any worker, running or finished, tagged with the
  worker it came from — so the view never falls back to poorer results once the best worker is done.

The view feeds ordinary reporters through their snapshot interface (`show_*`), so silent, bar and
tabular all render a parallel solve through the same code as a single one.
"""

from time import perf_counter

from max_div._core.solver._duration import Progress
from max_div._core.solver._progress_reporting import ProgressReporter, ProgressSnapshot

from ._result import WorkerResult

# The composite bar's resolution: progress fractions map onto this many tqdm steps.
_TQDM_N_TOTAL = 1000


class ParallelProgressView:
    """The view tracks every worker's snapshots and renders one composite stream through a reporter."""

    def __init__(self, reporter: ProgressReporter, n_workers: int) -> None:
        """Render through `reporter`, which decides throttling and layout; the view only composes.

        :param reporter: an ordinary rendering reporter (a tabular one laid out with worker columns).
        :param n_workers: how many workers the portfolio runs; indices `0..n_workers-1` are tracked.
        """
        self._reporter = reporter
        self._n_workers = n_workers
        self._t_start = perf_counter()
        self._latest: dict[int, ProgressSnapshot] = {}  # per worker, feeds the progress half
        self._best: ProgressSnapshot | None = None  # across workers, feeds the result half
        self._finished: set[int] = set()
        self._dead: set[int] = set()

    # -------------------------------------------------------------------------
    #  Events (called by the executor's drain loop)
    # -------------------------------------------------------------------------
    def start(self) -> None:
        """Start rendering; resets the wall clock the view stamps on every composite snapshot."""
        self._t_start = perf_counter()
        self._reporter.show_step_started(f"solving ({self._n_workers} workers)")

    def on_snapshot(self, snapshot: ProgressSnapshot) -> None:
        """Fold one worker's snapshot into the view and render the updated composite."""
        if snapshot.worker_index is None:
            return  # not attributable to a worker; nothing to fold it into
        self._latest[snapshot.worker_index] = snapshot
        if self._is_new_best(snapshot):
            self._best = snapshot
        self._reporter.show_update(self._composite())

    def on_worker_finished(self, result: WorkerResult) -> None:
        """Mark the worker finished and render its final state as an unthrottled milestone row."""
        self._finished.add(result.worker_index)
        final = self._latest.get(result.worker_index)
        if final is not None:
            self._reporter.show_milestone(self._composite(result_half=final))

    def on_worker_died(self, worker_index: int) -> None:
        """Drop a dead worker from the progress half, so the view keeps advancing without it."""
        self._dead.add(worker_index)

    def finish(self) -> None:
        """Render the closing composite row; called once, after the last worker reported or died."""
        if self._best is not None:
            self._reporter.show_step_finished(self._composite())

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _is_new_best(self, snapshot: ProgressSnapshot) -> bool:
        """Return whether this snapshot takes the result half: better score, ties to the lowest worker."""
        if self._best is None:
            return True
        if snapshot.worker_index == self._best.worker_index:
            return True  # newer state of the same worker; its score never regresses
        candidate = (snapshot.score, -snapshot.worker_index)  # ty: ignore[unsupported-operator]  # worker_index is set on every forwarded snapshot
        incumbent = (self._best.score, -self._best.worker_index)  # ty: ignore[unsupported-operator]  # (same)
        return candidate > incumbent

    def _composite(self, result_half: ProgressSnapshot | None = None) -> ProgressSnapshot:
        """Compose min-progress across live workers with the best (or given) result into one snapshot."""
        result = result_half if (result_half is not None) else self._best
        assert result is not None  # every caller renders only after at least one snapshot arrived

        # --- progress half: the slowest live worker ------
        live = [i for i in range(self._n_workers) if (i not in self._finished) and (i not in self._dead)]
        if live:
            fraction, iter_count = min(self._progress_of(i) for i in live)
        else:
            fraction, iter_count = 1.0, min((self._progress_of(i)[1] for i in self._finished), default=0)
        t_elapsed = perf_counter() - self._t_start
        progress = Progress(
            tqdm_n_total=_TQDM_N_TOTAL,
            fraction=fraction,
            iter_count=iter_count,
            est_n_iters_remaining=0,
            est_iters_per_second=0.0,
        )

        # --- result half: the best worker so far ---------
        return ProgressSnapshot(
            step_name="",
            progress=progress,
            t_elapsed_solver=t_elapsed,
            t_elapsed_step=t_elapsed,
            score=result.score,
            n_selected=result.n_selected,
            k=result.k,
            m=result.m,
            selection=None,
            ignore_infeasible_diversity=result.ignore_infeasible_diversity,
            selection_hash=result.selection_hash,
            debug_info=result.debug_info,
            worker_index=result.worker_index,
            n_active=len(live),
            worker_finished=result.worker_index in self._finished,
        )

    def _progress_of(self, worker_index: int) -> tuple[float, int]:
        """Return a worker's (fraction, iteration count), zero before its first snapshot arrives."""
        snapshot = self._latest.get(worker_index)
        if (snapshot is None) or (snapshot.progress is None):
            return 0.0, 0
        return snapshot.progress.fraction, snapshot.progress.iter_count
