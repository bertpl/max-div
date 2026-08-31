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
import traceback
from collections.abc import Sequence
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue

from max_div._core.metrics._distance import SharedStoreSpec, attached_distance_store
from max_div._core.solver._progress_reporting import ProgressReporter, ProgressSnapshot, SnapshotRequirements
from max_div._core.solver._solver_config import SolverConfig

from ._coordinator import WorkerCoordinator
from ._progress_channel import ForwardingProgressReporter
from ._progress_view import ParallelProgressView
from ._result import WorkerFailure, WorkerResult

# The drain loop wakes this often to re-check liveness while the queue is empty.  Messages that
# arrive are handled at once; this only bounds the wait after the last worker dies before that is
# noticed — short enough to notice fast, long enough that the poll costs nothing.  Not tied to
# solver runtime.
_POLL_SECONDS = 0.2

# Grace for a worker to exit on its own after reporting — normally immediate, since it just closes
# its shared-memory view and returns.  This only bounds the wait before force-terminating one that
# hangs in teardown; generous because it is off the critical path, and unrelated to solver runtime.
_JOIN_SECONDS = 30.0


def run_workers(
    configs: list[SolverConfig],
    spec: SharedStoreSpec,
    coordinators: Sequence[WorkerCoordinator],
    progress_reporter: ProgressReporter | None = None,
) -> tuple[list[WorkerResult], list[WorkerFailure]]:
    """Solve one configuration per worker over the published store, and return what each reported.

    Deciding what a failure means — warn, raise — is the caller's policy; `best_result`
    raises when no result came back at all.

    Args:
        configs: one solver configuration per worker, in worker order.
        spec: where the published store lives; every worker attaches to it.
        coordinators: one coordinator per worker, in worker order; `_coordinator` documents
            the topology this list wires up.
        progress_reporter: renders the workers' combined progress from this (parent) process; a
            reporter that renders nothing — or `None` — turns all forwarding off.

    Returns:
        `(results, failures)`, each in worker order, not arrival order, so the caller sees the
        same lists whichever worker happens to finish first.  A worker whose solve raises
        reports a `WorkerFailure`; one that dies without reporting anything (a hard kill)
        appears in neither list.

    Raises:
        ValueError: If the coordinator count does not match the worker count.
    """
    if len(coordinators) != len(configs):
        raise ValueError(f"Expected one coordinator per worker: got {len(coordinators)} for {len(configs)} workers.")
    requirements = progress_reporter.snapshot_requirements if (progress_reporter is not None) else None
    view = ParallelProgressView(progress_reporter, len(configs)) if (requirements is not None) else None  # ty: ignore[invalid-argument-type]  # requirements imply a reporter

    context = multiprocessing.get_context("spawn")
    messages: Queue = context.Queue()
    workers = [
        context.Process(
            target=solve_in_worker,
            args=(index, config, spec, coordinators[index], messages, requirements),
            daemon=True,
        )
        for index, config in enumerate(configs)
    ]
    if view is not None:
        view.start()
    for worker in workers:
        worker.start()
    try:
        collected, failures = _drain(messages, workers, view)
        if view is not None:
            view.finish()
    finally:
        _shut_down(workers)
    return (
        sorted(collected, key=lambda result: result.worker_index),
        sorted(failures, key=lambda failure: failure.worker_index),
    )


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
    try:
        with attached_distance_store(spec) as store:
            solution = config.build_solver(store=store).solve(coordinator=coordinator, progress_reporter=reporter)
            messages.put(WorkerResult(worker_index=worker_index, seed=config.seed, solution=solution))
    except Exception as exc:  # noqa: BLE001 -- report ANY failure to the parent
        # the exception is suppressed after reporting: re-raising would print the traceback to
        # this worker's own stderr, interleaving with the parent's live progress view
        messages.put(WorkerFailure(worker_index=worker_index, error=repr(exc), traceback_text=traceback.format_exc()))


def _drain(
    messages: Queue, workers: Sequence[BaseProcess], view: ParallelProgressView | None
) -> tuple[list[WorkerResult], list[WorkerFailure]]:
    """Handle messages until every worker has reported (a result or a failure) or none is left alive."""
    collected: list[WorkerResult] = []
    failures: list[WorkerFailure] = []
    reported_dead: set[int] = set()
    while len(collected) + len(failures) < len(workers):
        try:
            _handle(messages.get(timeout=_POLL_SECONDS), collected, failures, reported_dead, view)
        except queue_module.Empty:
            _notice_dead_workers(workers, collected, reported_dead, view)
            if not any(worker.is_alive() for worker in workers):
                # one last look: a worker can exit with messages still in flight through the queue
                _drain_remaining(messages, collected, failures, reported_dead, view)
                break
    return collected, failures


def _handle(
    message: "ProgressSnapshot | WorkerResult | WorkerFailure",
    collected: list[WorkerResult],
    failures: list[WorkerFailure],
    reported_dead: set[int],
    view: ParallelProgressView | None,
) -> None:
    """Fold one message into the results (a result), the failures, or the rendered view (a snapshot)."""
    if isinstance(message, WorkerResult):
        collected.append(message)
        if view is not None:
            view.on_worker_finished(message)
    elif isinstance(message, WorkerFailure):
        failures.append(message)
        # mark the worker dead in the view now, and in `reported_dead` so the liveness poll
        # does not report it a second time once the process actually exits
        if message.worker_index not in reported_dead:
            reported_dead.add(message.worker_index)
            if view is not None:
                view.on_worker_died(message.worker_index)
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
    messages: Queue,
    collected: list[WorkerResult],
    failures: list[WorkerFailure],
    reported_dead: set[int],
    view: ParallelProgressView | None,
) -> None:
    """Handle whatever reports are still in flight, giving each a short wait before giving up.

    Every worker is dead by the time this runs, so the queue can only shrink; the loop ends at
    the first empty poll.
    """
    while True:
        try:
            _handle(messages.get(timeout=_POLL_SECONDS), collected, failures, reported_dead, view)
        except queue_module.Empty:
            break


def _shut_down(workers: Sequence[BaseProcess]) -> None:
    """Wait for each worker to exit, killing any that will not."""
    for worker in workers:
        worker.join(timeout=_JOIN_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=_JOIN_SECONDS)
