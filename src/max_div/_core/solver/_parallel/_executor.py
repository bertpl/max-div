"""One solver runs per worker process over a single shared store, and the executor collects the results.

Workers are **spawned, never forked**.  The parent runs numba parallel code while building the
distance store, and numba's threading layer does not survive a fork — a forked child deadlocks on
its first parallel call.

Each worker is a process rather than a thread because the search is Python-level and would contend
on the interpreter lock.  Only the distances are shared; every worker allocates its own bookkeeping,
which is small next to the distances.

One queue carries everything the workers send — progress snapshots while they solve, a result each
when they finish — and the parent drains it in a render loop that runs *while* the workers solve.
Workers never write to their own stdout; the parent owns the only terminal writer.  The queue is
deliberately unbounded (see `_progress_channel` for why a bounded one deadlocks on a dying worker).
"""

import multiprocessing
import queue as queue_module
from collections.abc import Sequence
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue

from max_div._core.metrics._distance import SharedStoreSpec, attached_distance_store
from max_div._core.solver._progress_reporting import ProgressReporter, ProgressSnapshot, SnapshotRequirements
from max_div._core.solver._solver_config import SolverConfig

from ._coordinator import WorkerCoordinator
from ._progress_channel import ForwardingProgressReporter
from ._progress_view import ParallelProgressView
from ._result import WorkerResult

# How often the drain loop wakes to re-check liveness while the queue is empty.  Messages that
# arrive are handled at once; this only bounds the wait after the last worker dies before that is
# noticed — short enough to notice fast, long enough that the poll costs nothing.  Not tied to
# solver runtime.
_POLL_SECONDS = 0.2

# Grace for a worker to exit on its own after reporting — normally immediate, since it just closes
# its shared-memory view and returns.  This only bounds the wait before force-terminating one that
# hangs in teardown; generous because it is off the critical path, and unrelated to solver runtime.
_JOIN_SECONDS = 30.0


def run_portfolio(
    configs: list[SolverConfig],
    spec: SharedStoreSpec,
    coordinator: WorkerCoordinator,
    progress_reporter: ProgressReporter | None = None,
) -> list[WorkerResult]:
    """Solve one configuration per worker over the published store, and return what each reported.

    Results come back in worker order rather than arrival order, so the caller sees the same list
    whichever worker happens to finish first.  A worker that dies without reporting is left out rather than
    treated as an error; `best_result` raises when none came back.

    :param configs: one solver configuration per worker, in worker order.
    :param spec: where the published store lives; every worker attaches to it.
    :param coordinator: reached by every worker at each batch boundary; an independent one shares nothing.
    :param progress_reporter: renders the workers' combined progress from this (parent) process; a
                              reporter that renders nothing — or `None` — turns all forwarding off.
    """
    requirements = progress_reporter.snapshot_requirements if (progress_reporter is not None) else None
    view = ParallelProgressView(progress_reporter, len(configs)) if (requirements is not None) else None  # ty: ignore[invalid-argument-type]  # requirements imply a reporter

    context = multiprocessing.get_context("spawn")
    messages: Queue = context.Queue()
    workers = [
        context.Process(
            target=solve_in_worker, args=(index, config, spec, coordinator, messages, requirements), daemon=True
        )
        for index, config in enumerate(configs)
    ]
    if view is not None:
        view.start()
    for worker in workers:
        worker.start()
    try:
        collected = _drain(messages, workers, view)
        if view is not None:
            view.finish()
    finally:
        _shut_down(workers)
    return sorted(collected, key=lambda result: result.worker_index)


def solve_in_worker(
    worker_index: int,
    config: SolverConfig,
    spec: SharedStoreSpec,
    coordinator: WorkerCoordinator,
    messages: Queue,
    requirements: SnapshotRequirements | None,
) -> None:
    """Solve one configuration in this process and report the result, then release the store.

    This function is the entry point of a spawned worker, so it must stay importable by name — a
    spawned child reconstructs the function from its module path rather than inheriting it.
    """
    if requirements is not None:
        reporter: ProgressReporter = ForwardingProgressReporter(messages, worker_index, requirements)
    else:
        reporter = ProgressReporter.silent()
    with attached_distance_store(spec) as store:
        solution = config.build_solver(store).solve(coordinator=coordinator, progress_reporter=reporter)
        messages.put(WorkerResult(worker_index=worker_index, seed=config.seed, solution=solution))


def _drain(messages: Queue, workers: Sequence[BaseProcess], view: ParallelProgressView | None) -> list[WorkerResult]:
    """Handle messages until every worker has reported or none is left alive, rendering as they come."""
    collected: list[WorkerResult] = []
    reported_dead: set[int] = set()
    while len(collected) < len(workers):
        try:
            _handle(messages.get(timeout=_POLL_SECONDS), collected, view)
        except queue_module.Empty:
            _notice_dead_workers(workers, collected, reported_dead, view)
            if not any(worker.is_alive() for worker in workers):
                # one last look: a worker can exit with messages still in flight through the queue
                _drain_remaining(messages, collected, view, limit=len(workers) - len(collected))
                break
    return collected


def _handle(
    message: "ProgressSnapshot | WorkerResult", collected: list[WorkerResult], view: ParallelProgressView | None
) -> None:
    """Fold one message into the collection (a result) or the rendered view (a snapshot)."""
    if isinstance(message, WorkerResult):
        collected.append(message)
        if view is not None:
            view.on_worker_finished(message)
    elif view is not None:
        view.on_snapshot(message)


def _notice_dead_workers(
    workers: Sequence[BaseProcess],
    collected: list[WorkerResult],
    reported_dead: set[int],
    view: ParallelProgressView | None,
) -> None:
    """Tell the view about workers that stopped without reporting, so the view advances without them."""
    if view is None:
        return
    indices_with_result = {result.worker_index for result in collected}
    for index, worker in enumerate(workers):
        if (not worker.is_alive()) and (index not in indices_with_result) and (index not in reported_dead):
            reported_dead.add(index)
            view.on_worker_died(index)


def _drain_remaining(
    messages: Queue, collected: list[WorkerResult], view: ParallelProgressView | None, limit: int
) -> None:
    """Handle up to `limit` more results, giving each a short wait before giving up."""
    while limit > 0:
        try:
            n_results_before = len(collected)
            _handle(messages.get(timeout=_POLL_SECONDS), collected, view)
            limit -= len(collected) - n_results_before  # snapshots don't count toward the limit
        except queue_module.Empty:
            break


def _shut_down(workers: Sequence[BaseProcess]) -> None:
    """Wait for each worker to exit, killing any that will not."""
    for worker in workers:
        worker.join(timeout=_JOIN_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=_JOIN_SECONDS)
