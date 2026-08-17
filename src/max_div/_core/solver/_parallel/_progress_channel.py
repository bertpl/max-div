"""A worker-side reporter forwards progress snapshots to the parent process instead of printing.

Workers never write to their own stdout — that invariant is what keeps N workers' progress from
interleaving. Each worker's solver reports into a `ForwardingProgressReporter`, which materializes
the snapshot (a snapshot built in-process holds the selection by reference and a debug *callable*;
neither can cross a process boundary) and puts it on the queue the parent drains.

The queue must be unbounded: with a bounded queue, a worker that dies hard leaves the slots held by
its unfed buffer unreleasable, the queue reads full forever, and the surviving workers wedge on
their own puts. Backpressure is therefore the throttle's job — it bounds traffic to a few snapshots
per second per worker by construction.
"""

from collections.abc import Callable
from dataclasses import replace
from multiprocessing.queues import Queue

from max_div._core.solver._progress_reporting import (
    TABULAR_C_SLOWDOWNS,
    ProgressReporter,
    ProgressSnapshot,
    ReportThrottle,
    SnapshotRequirements,
    selection_hash_str,
)

# The forwarding cadence is the fastest any rendering reporter uses — derived, so it stays that by
# construction — and the parent's own display throttle, not this one, decides the visible row
# spacing at every verbosity level.
_FORWARD_C_SLOWDOWN = min(TABULAR_C_SLOWDOWNS.values())


class ForwardingProgressReporter(ProgressReporter):
    """A forwarding reporter puts materialized snapshots onto a queue instead of rendering them."""

    def __init__(self, queue: Queue, worker_index: int, requirements: SnapshotRequirements) -> None:
        """Forward this worker's snapshots, materializing what the parent's reporter declared it needs.

        Args:
            queue: the parent-drained progress queue; must be unbounded (see module docstring).
            worker_index: stamped on every forwarded snapshot, so the parent can attribute it.
            requirements: what to materialize, as declared by the parent's rendering reporter.
        """
        super().__init__()
        self._queue = queue
        self._worker_index = worker_index
        self._requirements = requirements
        self._throttle = ReportThrottle(c_slowdown=_FORWARD_C_SLOWDOWN)

    # -------------------------------------------------------------------------
    #  Rendering interface (forwards instead of rendering)
    # -------------------------------------------------------------------------
    def show_step_started(self, step_name: str) -> None:
        self._throttle.reset()

    def show_update(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        iter_now = snapshot.progress.iter_count if (snapshot.progress is not None) else 0
        if self._throttle.passes(iter_now, snapshot.t_elapsed_step):
            self._queue.put(self._materialize(snapshot, get_debug_info))

    def show_step_finished(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        # step ends are never throttled: they are rare, and the last one carries the worker's final state
        self._queue.put(self._materialize(snapshot, get_debug_info))

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _materialize(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None) -> ProgressSnapshot:
        """Return a picklable copy: by-reference fields resolved or dropped, worker index stamped."""
        return replace(
            snapshot,
            selection=None,
            selection_hash=selection_hash_str(snapshot) if self._requirements.selection_hash else None,
            debug_info=(
                get_debug_info()
                if (self._requirements.debug_info and (get_debug_info is not None))
                else snapshot.debug_info
            ),
            worker_index=self._worker_index,
            n_selected=int(snapshot.n_selected),
            k=int(snapshot.k),
            m=int(snapshot.m),
        )
