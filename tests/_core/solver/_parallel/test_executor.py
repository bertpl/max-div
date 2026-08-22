import multiprocessing
import queue

import numpy as np
import pytest

from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder
from max_div._core.solver._distance_storage import build_shared_distance_store
from max_div._core.solver._duration import iterations
from max_div._core.solver._parallel import (
    CooperativeCoordinator,
    GroupIncumbentSlot,
    IndependentCoordinator,
    best_result,
    run_workers,
)
from max_div._core.solver._parallel._executor import _drain, _notice_dead_workers, solve_in_worker
from max_div._core.solver._parallel._progress_view import ParallelProgressView
from max_div._core.solver._parallel._result import WorkerResult
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._progress_reporting import ProgressReporter, SnapshotRequirements, Verbosity

_SEEDS = (11, 22, 33)


def _builder() -> MaxDivSolverBuilder:
    """Return a builder over a problem small enough to solve several times in a test."""
    rng = np.random.default_rng(20260809)
    problem = MaxDivProblem.new(rng.random((60, 3)).astype(np.float32), k=6)
    return MaxDivSolverBuilder(problem).with_preset(iterations(50), SolverPreset.SMART)


@pytest.fixture
def parallel_results():
    """Run one set of spawned workers, and hand back their results with the builder used."""
    builder = _builder()
    resolved, config = builder.prepare_storage_and_config()
    with build_shared_distance_store(builder._problem, resolved) as shared:
        yield (
            builder,
            run_workers(
                [config.with_seed(seed) for seed in _SEEDS], shared.spec, [IndependentCoordinator() for _ in _SEEDS]
            ),
        )


def test_every_worker_reports_a_result(parallel_results):
    """A parallel solve returns one result per configuration, each carrying a full selection."""
    # --- arrange / act ----------------
    builder, results = parallel_results

    # --- assert -----------------------
    assert len(results) == len(_SEEDS)
    assert all(result.i_selected.size == builder._k for result in results)


def test_results_come_back_in_worker_order(parallel_results):
    """Results are ordered by worker rather than by who finished first."""
    # --- arrange / act ----------------
    _, results = parallel_results

    # --- assert -----------------------
    assert [result.worker_index for result in results] == list(range(len(_SEEDS)))
    assert [result.seed for result in results] == list(_SEEDS)


def test_a_worker_reproduces_the_same_solve_run_alone(parallel_results):
    """A worker's selection is what the same configuration and seed produce in a single process."""
    # --- arrange ----------------------
    builder, results = parallel_results

    # --- act --------------------------
    alone = builder.with_seed(_SEEDS[0]).build().solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    np.testing.assert_array_equal(np.sort(results[0].i_selected), np.sort(alone.i_selected))


def test_the_best_reported_result_wins(parallel_results):
    """The winner is the best-scoring worker, not the first to report."""
    # --- arrange / act ----------------
    _, results = parallel_results
    winner = best_result(results)

    # --- assert -----------------------
    assert winner.score == max(result.score for result in results)


def test_one_coordinator_per_worker_is_required():
    """A coordinator count that does not match the worker count is rejected before any worker spawns."""
    # --- arrange ----------------------
    builder = _builder()
    resolved, config = builder.prepare_storage_and_config()

    # --- act & assert -----------------
    with build_shared_distance_store(builder._problem, resolved) as shared, pytest.raises(ValueError):
        run_workers([config.with_seed(1), config.with_seed(2)], shared.spec, [IndependentCoordinator()])


def test_a_group_of_cooperative_workers_solves_and_exchanges():
    """Spawned workers sharing one incumbent slot all report results, and the slot was published to."""
    # --- arrange ----------------------
    builder = _builder()
    resolved, config = builder.prepare_storage_and_config()
    n_score_components = 3 + len(config.diversity_tie_breakers)
    slot = GroupIncumbentSlot(multiprocessing.get_context("spawn"), k=builder._k, score_length=n_score_components)
    coordinators = [CooperativeCoordinator(slot) for _ in _SEEDS]

    # --- act --------------------------
    with build_shared_distance_store(builder._problem, resolved) as shared:
        results = run_workers([config.with_seed(seed) for seed in _SEEDS], shared.spec, coordinators)

    # --- assert -----------------------
    assert len(results) == len(_SEEDS)
    assert all(result.i_selected.size == builder._k for result in results)
    assert slot.written  # at least the first boundary reached published into the empty slot


def test_parallel_solve_renders_coherent_progress(capsys):
    """A rendered parallel solve prints one non-interleaved table and still collects every result."""
    # --- arrange ----------------------
    builder = _builder()
    resolved, config = builder.prepare_storage_and_config()
    reporter = ProgressReporter.from_verbosity(Verbosity.TABULAR, worker_columns=True)

    # --- act --------------------------
    with build_shared_distance_store(builder._problem, resolved) as shared:
        results = run_workers(
            [config.with_seed(seed) for seed in _SEEDS],
            shared.spec,
            [IndependentCoordinator() for _ in _SEEDS],
            progress_reporter=reporter,
        )

    # --- assert -----------------------
    assert len(results) == len(_SEEDS)
    lines = capsys.readouterr().out.splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    assert lines == [line for line in lines if line.startswith("|")]  # nothing but table lines: no interleaving
    assert "Worker" in table_lines[0]
    assert "Active" in table_lines[0]
    assert sum("✓" in line for line in table_lines) >= len(_SEEDS)  # every worker got its finishing row


def test_solve_in_worker_runs_in_process():
    """The worker entry point solves and reports, with and without a forwarding reporter."""
    # --- arrange ----------------------
    builder = _builder()
    resolved, config = builder.prepare_storage_and_config()
    messages = queue.Queue()
    requirements = SnapshotRequirements(debug_info=False, selection_hash=True)

    # --- act --------------------------
    with build_shared_distance_store(builder._problem, resolved) as shared:
        solve_in_worker(0, config.with_seed(1), shared.spec, IndependentCoordinator(), messages, None)
        solve_in_worker(1, config.with_seed(2), shared.spec, IndependentCoordinator(), messages, requirements)

    # --- assert -----------------------
    received = []
    while not messages.empty():
        received.append(messages.get_nowait())
    results = [message for message in received if isinstance(message, WorkerResult)]
    snapshots = [message for message in received if not isinstance(message, WorkerResult)]
    assert [result.worker_index for result in results] == [0, 1]
    assert len(snapshots) > 0  # only the forwarding-reporter run produced snapshots
    assert all(snapshot.worker_index == 1 for snapshot in snapshots)


class _StubWorker:
    """A stub worker stands in for a worker process, answering liveness checks with a fixed value."""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def test_drain_collects_in_flight_results_of_dead_workers():
    """Results still in the queue after every worker exited are collected, not lost."""
    # --- arrange ----------------------
    builder = _builder()
    resolved, config = builder.prepare_storage_and_config()
    messages = queue.Queue()
    with build_shared_distance_store(builder._problem, resolved) as shared:
        solve_in_worker(0, config.with_seed(1), shared.spec, IndependentCoordinator(), messages, None)
    workers = [_StubWorker(alive=False), _StubWorker(alive=False)]  # worker 1 died without reporting

    # --- act --------------------------
    collected = _drain(messages, workers, view=None)

    # --- assert -----------------------
    assert [result.worker_index for result in collected] == [0]


def test_dead_workers_are_reported_to_the_view_once():
    """A worker that stopped without a result is reported dead to the view, exactly once."""
    # --- arrange ----------------------
    reporter = ProgressReporter.from_verbosity(Verbosity.TABULAR, worker_columns=True)
    view = ParallelProgressView(reporter, n_workers=2)
    workers = [_StubWorker(alive=True), _StubWorker(alive=False)]
    reported_dead: set[int] = set()

    # --- act --------------------------
    _notice_dead_workers(workers, [], reported_dead, view)
    _notice_dead_workers(workers, [], reported_dead, view)

    # --- assert -----------------------
    assert reported_dead == {1}
