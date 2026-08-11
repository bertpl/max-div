import queue

import numpy as np

from max_div._core.solver._duration import Progress
from max_div._core.solver._parallel._progress_channel import ForwardingProgressReporter
from max_div._core.solver._progress_reporting import ProgressSnapshot, SnapshotRequirements
from max_div._core.solver._score import Score


def _snapshot(iter_count: int = 5, t_elapsed: float = 1.0) -> ProgressSnapshot:
    """Return an in-process snapshot with by-reference fields set, as the base reporter builds them."""
    return ProgressSnapshot(
        step_name="step 1",
        progress=Progress(
            tqdm_n_total=100,
            fraction=0.5,
            iter_count=iter_count,
            est_n_iters_remaining=5,
            est_iters_per_second=1.0,
        ),
        t_elapsed_solver=t_elapsed,
        t_elapsed_step=t_elapsed,
        score=Score(size=1.0, constraints=1.0, diversity=0.5, div_tie_breakers=()),
        n_selected=np.int32(3),
        k=np.int32(5),
        m=np.int32(0),
        selection=np.arange(3, dtype=np.int32),
        ignore_infeasible_diversity=False,
    )


def test_forwarded_snapshot_is_materialized():
    """Forwarding resolves by-reference fields into picklable ones and stamps the worker index."""
    # --- arrange -----------------------------------------
    messages = queue.Queue()
    reporter = ForwardingProgressReporter(
        messages, worker_index=3, requirements=SnapshotRequirements(debug_info=True, selection_hash=True)
    )

    # --- act ---------------------------------------------
    reporter.show_step_started("step 1")
    reporter.show_update(_snapshot(), get_debug_info=lambda: "dbg")

    # --- assert ------------------------------------------
    forwarded = messages.get_nowait()
    assert forwarded.worker_index == 3
    assert forwarded.selection is None
    assert isinstance(forwarded.selection_hash, str) and len(forwarded.selection_hash) > 0
    assert forwarded.debug_info == "dbg"
    assert type(forwarded.n_selected) is int and type(forwarded.k) is int and type(forwarded.m) is int
    assert forwarded.score == _snapshot().score


def test_forwarding_skips_fields_not_required():
    """Neither the hash nor the debug string is produced when the requirements exclude them."""
    # --- arrange -----------------------------------------
    messages = queue.Queue()
    reporter = ForwardingProgressReporter(
        messages, worker_index=0, requirements=SnapshotRequirements(debug_info=False, selection_hash=False)
    )

    def _boom() -> str:
        raise AssertionError("debug callable must not be invoked when not required")

    # --- act ---------------------------------------------
    reporter.show_step_started("step 1")
    reporter.show_update(_snapshot(), get_debug_info=_boom)

    # --- assert ------------------------------------------
    forwarded = messages.get_nowait()
    assert forwarded.selection_hash is None
    assert forwarded.debug_info is None


def test_forwarding_is_throttled_but_step_finished_is_not():
    """Rapid-fire updates collapse under the throttle, while every step end goes through."""
    # --- arrange -----------------------------------------
    messages = queue.Queue()
    reporter = ForwardingProgressReporter(
        messages, worker_index=0, requirements=SnapshotRequirements(debug_info=False, selection_hash=False)
    )

    # --- act ---------------------------------------------
    reporter.show_step_started("step 1")
    for i in range(50):
        reporter.show_update(_snapshot(iter_count=i, t_elapsed=i * 0.01))  # 50 updates in a fast 0.5s burst
    for _ in range(3):
        reporter.show_step_finished(_snapshot(t_elapsed=0.5))

    # --- assert ------------------------------------------
    n_forwarded = messages.qsize()
    assert n_forwarded < (50 + 3) / 2  # the update burst was thinned to (well below) half...
    assert n_forwarded >= 1 + 3  # ...but the first update and every step end went through
