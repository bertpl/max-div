"""One solver runs per worker process over a single shared store, and the executor collects the results.

Workers are **spawned, never forked**.  The parent runs numba parallel code while building the
distance store, and numba's threading layer does not survive a fork — a forked child deadlocks on
its first parallel call.

Each worker is a process rather than a thread because the search is Python-level and would contend
on the interpreter lock.  Only the distances are shared; every worker allocates its own bookkeeping,
which is small next to the distances.
"""

import multiprocessing
import queue as queue_module
from collections.abc import Sequence
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue

from max_div._core.metrics._distance import SharedStoreSpec, attached_distance_store
from max_div._core.solver._solver_config import SolverConfig

from ._coordinator import WorkerCoordinator
from ._result import WorkerResult

# How long to wait on the result queue before checking again whether any worker is still alive.
_POLL_SECONDS = 0.2

# How long to wait for a worker that has reported its result to exit on its own.
_JOIN_SECONDS = 30.0


def run_portfolio(
    configs: list[SolverConfig], spec: SharedStoreSpec, coordinator: WorkerCoordinator
) -> list[WorkerResult]:
    """Solve one configuration per worker over the published store, and return what each reported.

    Returns in worker order rather than arrival order, so the caller sees the same list whichever
    worker happens to finish first.  A worker that dies without reporting is left out rather than
    fatal; `best_result` raises when none came back.

    :param configs: one solver configuration per worker, in worker order.
    :param spec: the published store every worker attaches to.
    :param coordinator: the `WorkerCoordinator` handed to every worker.
    """
    context = multiprocessing.get_context("spawn")
    results: Queue = context.Queue()
    workers = [
        context.Process(target=solve_in_worker, args=(index, config, spec, coordinator, results), daemon=True)
        for index, config in enumerate(configs)
    ]
    for worker in workers:
        worker.start()
    try:
        collected = _collect(results, workers)
    finally:
        _shut_down(workers)
    return sorted(collected, key=lambda result: result.worker_index)


def solve_in_worker(
    worker_index: int,
    config: SolverConfig,
    spec: SharedStoreSpec,
    coordinator: WorkerCoordinator,
    results: Queue,
) -> None:
    """Solve one configuration in this process and report the result, then release the store.

    The entry point of a spawned worker, so it must stay importable by name — a spawned child
    reconstructs it from the module path rather than inheriting it.
    """
    with attached_distance_store(spec) as store:
        solution = config.build_solver(store).solve(verbosity=0, coordinator=coordinator)
        results.put(
            WorkerResult(
                worker_index=worker_index,
                i_selected=solution.i_selected,
                score=solution.score,
                elapsed=solution.duration,
                seed=config.seed,
            )
        )


def _collect(results: Queue, workers: Sequence[BaseProcess]) -> list[WorkerResult]:
    """Take results off the queue until every worker has reported or none is left alive."""
    collected: list[WorkerResult] = []
    while len(collected) < len(workers):
        try:
            collected.append(results.get(timeout=_POLL_SECONDS))
        except queue_module.Empty:
            if not any(worker.is_alive() for worker in workers):
                # one last look: a worker can exit with its result still in flight through the queue
                collected.extend(_drain(results, limit=len(workers) - len(collected)))
                break
    return collected


def _drain(results: Queue, limit: int) -> list[WorkerResult]:
    """Take up to `limit` more results, giving each a short wait before giving up."""
    drained: list[WorkerResult] = []
    for _ in range(limit):
        try:
            drained.append(results.get(timeout=_POLL_SECONDS))
        except queue_module.Empty:
            break
    return drained


def _shut_down(workers: Sequence[BaseProcess]) -> None:
    """Wait for each worker to exit, killing any that will not."""
    for worker in workers:
        worker.join(timeout=_JOIN_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=_JOIN_SECONDS)
