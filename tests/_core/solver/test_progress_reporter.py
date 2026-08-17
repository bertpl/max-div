from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from max_div._core.solver._duration import Progress
from max_div._core.solver._progress_reporting import (
    ProgressReporter,
    ProgressSnapshot,
    ReportThrottle,
    SilentProgressReporter,
    SnapshotRequirements,
    TabularProgressReporter,
    TqdmProgressReporter,
    Verbosity,
)
from max_div._core.solver._score import Score


def _stub_state(n_selected: int = 3, k: int = 5, m: int = 2) -> SimpleNamespace:
    """Return a stand-in for SolverState carrying just the fields snapshot building reads."""
    return SimpleNamespace(
        score=Score(size=1.0, constraints=0.5, diversity=0.25, div_tie_breakers=()),
        n_selected=n_selected,
        k=k,
        m=m,
        selected_index_array=np.arange(n_selected, dtype=np.int32),
    )


def _stub_progress(iter_count: int = 7) -> Progress:
    """Return a Progress with fixed, easily recognizable values."""
    return Progress(
        tqdm_n_total=100,
        fraction=0.42,
        iter_count=iter_count,
        est_n_iters_remaining=10,
        est_iters_per_second=2.0,
    )


class _RecordingProgressReporter(ProgressReporter):
    """A reporter that renders nothing and records every show_* call, so tests can inspect the snapshots built."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, object]] = []

    def show_step_started(self, step_name: str) -> None:
        self.calls.append(("started", step_name))

    def show_update(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        self.calls.append(("update", snapshot))

    def show_step_finished(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        self.calls.append(("finished", snapshot))


@pytest.mark.parametrize(
    "factory_method, expected_class",
    [
        (ProgressReporter.silent, SilentProgressReporter),
        (ProgressReporter.tqdm, TqdmProgressReporter),
        (ProgressReporter.tabular, TabularProgressReporter),
    ],
)
def test_progress_reporter_factory_methods(factory_method: Callable, expected_class: type[ProgressReporter]):
    # --- act --------------------------
    reporter = factory_method()

    # --- assert -----------------------
    assert isinstance(reporter, expected_class)


@pytest.mark.parametrize(
    "selection, n",
    [
        (np.array([1, 2, 100, 516], dtype=np.int32), 20),
        (np.array([1, 2, 100, 516, 700], dtype=np.int32), 30),
        (np.arange(1000, dtype=np.int32), 40),
    ],
)
def test_progress_reporter_selection_hash(selection: np.ndarray, n: int):
    # --- arrange ----------------------
    selection_modified = selection.copy()
    selection_modified[-1] += 1  # modify selection to ensure hash changes

    # --- act --------------------------
    hash_str_1 = TabularProgressReporter._get_selection_hash(selection, n)
    hash_str_2 = TabularProgressReporter._get_selection_hash(selection_modified, n)

    # --- assert -----------------------
    assert len(hash_str_1) == n
    assert all(char in "0123456789abcdef" for char in hash_str_1)

    assert len(hash_str_2) == n
    assert all(char in "0123456789abcdef" for char in hash_str_2)

    assert hash_str_1 != hash_str_2
    assert hash_str_1[:8] != hash_str_2[:8]  # even first part should be different, if just the last input digit changed


@pytest.mark.parametrize(
    "verbosity, expected_class, expected_c_slowdown, expected_debug_info",
    [
        (0, SilentProgressReporter, None, None),
        (10, TqdmProgressReporter, None, None),
        (20, TabularProgressReporter, 1.10, False),
        (21, TabularProgressReporter, 1.05, False),
        (22, TabularProgressReporter, 1.02, False),
        (23, TabularProgressReporter, 1.01, False),
        (25, TabularProgressReporter, 1.01, True),
    ],
)
def test_from_verbosity(
    verbosity: int,
    expected_class: type[ProgressReporter],
    expected_c_slowdown: float | None,
    expected_debug_info: bool | None,
):
    """Each verbosity level maps to the expected reporter class and settings."""
    # --- act --------------------------
    reporter = ProgressReporter.from_verbosity(verbosity)

    # --- assert -----------------------
    assert type(reporter) is expected_class
    if expected_c_slowdown is not None:
        assert reporter._c_slowdown == expected_c_slowdown
        assert reporter._debug_info == expected_debug_info


@pytest.mark.parametrize("verbosity", [-1, 1, 11, 24, 26, 42])
def test_from_verbosity_invalid_level(verbosity: int):
    """An unknown verbosity level raises ValueError."""
    # --- act & assert -----------------
    with pytest.raises(ValueError):
        ProgressReporter.from_verbosity(verbosity)


def test_snapshot_building():
    """The base class builds snapshots carrying the step name, elapsed times, and the state fields."""
    # --- arrange ----------------------
    reporter = _RecordingProgressReporter()
    state = _stub_state(n_selected=3, k=5, m=2)
    progress = _stub_progress(iter_count=7)

    # --- act --------------------------
    reporter.solver_step_started("step A")
    reporter.update(progress, state, ignore_infeasible_diversity=True)
    reporter.solver_step_finished(None, state)

    # --- assert -----------------------
    assert [call[0] for call in reporter.calls] == ["started", "update", "finished"]

    snapshot = reporter.calls[1][1]
    assert isinstance(snapshot, ProgressSnapshot)
    assert snapshot.step_name == "step A"
    assert snapshot.progress is progress
    assert snapshot.score is state.score
    assert (snapshot.n_selected, snapshot.k, snapshot.m) == (3, 5, 2)
    assert snapshot.selection is state.selected_index_array  # by reference, not copied
    assert snapshot.ignore_infeasible_diversity is True
    assert snapshot.t_elapsed_solver >= snapshot.t_elapsed_step >= 0.0

    snapshot_finished = reporter.calls[2][1]
    assert snapshot_finished.progress is None
    assert snapshot_finished.ignore_infeasible_diversity is False


def test_snapshot_solver_clock_spans_steps():
    """The solver clock starts at the first step and keeps running across step boundaries."""
    # --- arrange ----------------------
    reporter = _RecordingProgressReporter()
    state = _stub_state()

    # --- act --------------------------
    reporter.solver_step_started("step A")
    reporter.solver_step_started("step B")
    reporter.update(_stub_progress(), state)

    # --- assert -----------------------
    snapshot = reporter.calls[-1][1]
    assert snapshot.step_name == "step B"
    assert snapshot.t_elapsed_solver >= snapshot.t_elapsed_step  # solver clock was not reset by step B


def test_tabular_show_update_without_progress(capsys):
    """A snapshot without progress renders a row with blank progress columns instead of crashing."""
    # --- arrange ----------------------
    reporter = TabularProgressReporter()
    state = _stub_state()

    # --- act --------------------------
    reporter.solver_step_started("step A")
    reporter.update(None, state)  # ty: ignore[invalid-argument-type]  # deliberately exercising the None path

    # --- assert -----------------------
    output_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("|")]
    row = output_lines[-1]
    assert "step A" in row
    assert "%" not in row  # progress columns are blank
    assert "3/     5" in row


def test_tqdm_show_update_without_progress():
    """A snapshot without progress leaves the tqdm bar untouched instead of crashing."""
    # --- arrange ----------------------
    reporter = TqdmProgressReporter()
    state = _stub_state()
    reporter.solver_step_started("step A")
    n_before = reporter._current_pbar.n

    # --- act --------------------------
    reporter.update(None, state)  # ty: ignore[invalid-argument-type]  # deliberately exercising the None path

    # --- assert -----------------------
    assert reporter._current_pbar.n == n_before
    reporter.solver_step_finished(None, state)


@pytest.mark.parametrize("verbosity", list(Verbosity))
def test_from_verbosity_accepts_enum_members(verbosity: Verbosity):
    """Every Verbosity member creates the same reporter as its plain integer value."""
    # --- act --------------------------
    from_member = ProgressReporter.from_verbosity(verbosity)
    from_int = ProgressReporter.from_verbosity(int(verbosity))

    # --- assert -----------------------
    assert type(from_member) is type(from_int)


@pytest.mark.parametrize(
    "reporter, expected",
    [
        (SilentProgressReporter(), None),
        (TqdmProgressReporter(), SnapshotRequirements(debug_info=False, selection_hash=False)),
        (TabularProgressReporter(), SnapshotRequirements(debug_info=False, selection_hash=True)),
        (TabularProgressReporter(debug_info=True), SnapshotRequirements(debug_info=True, selection_hash=True)),
    ],
)
def test_snapshot_requirements(reporter: ProgressReporter, expected: SnapshotRequirements | None):
    """Each reporter class declares what must be materialized for it — silent declares nothing at all."""
    # --- act & assert -----------------
    assert reporter.snapshot_requirements == expected


def test_report_throttle_thins_a_burst_and_resets():
    """A rapid burst collapses to a few passes; reset makes the next update pass again."""
    # --- arrange ----------------------
    throttle = ReportThrottle(c_slowdown=1.05)

    # --- act --------------------------
    n_passed = sum(throttle.passes(iter_now=i, t_elapsed=i * 0.01) for i in range(100))
    blocked_before_reset = not throttle.passes(iter_now=100, t_elapsed=1.0000001)
    throttle.reset()
    passes_after_reset = throttle.passes(iter_now=0, t_elapsed=0.0)

    # --- assert -----------------------
    assert 1 <= n_passed < 50
    assert blocked_before_reset
    assert passes_after_reset


def test_tabular_worker_columns_layout(capsys):
    """Worker columns replace the step columns: worker index (with finished marker) and active count."""
    # --- arrange ----------------------
    reporter = TabularProgressReporter(worker_columns=True)
    snapshot_running = ProgressSnapshot(
        step_name="",
        progress=_stub_progress(),
        t_elapsed_solver=1.0,
        t_elapsed_step=1.0,
        score=Score(size=1.0, constraints=1.0, diversity=0.5, div_tie_breakers=()),
        n_selected=5,
        k=5,
        m=0,
        selection=None,
        ignore_infeasible_diversity=False,
        selection_hash="cafe",
        worker_index=2,
        n_active=4,
    )
    snapshot_finished = replace(snapshot_running, worker_index=3, n_active=3, worker_finished=True)

    # --- act --------------------------
    reporter.show_step_started("solving (4 workers)")
    reporter.show_update(snapshot_running)
    reporter.show_milestone(snapshot_finished)

    # --- assert -----------------------
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("|")]
    header, running_row, milestone_row, closing_line = lines[0], lines[2], lines[3], lines[4]
    assert "Worker" in header
    assert "Active" in header
    assert "Solver step" not in header
    assert "Step %" not in header
    assert " 2 " in running_row
    assert "3✓" in milestone_row
    assert "cafe" in running_row
    assert set(closing_line.replace("|", "").replace(" ", "")) == {"-"}


def test_tabular_prefers_materialized_debug_info(capsys):
    """A pre-resolved debug string wins over the callable, which is how the parent renders forwarded snapshots."""
    # --- arrange ----------------------
    reporter = TabularProgressReporter(debug_info=True)
    state = _stub_state()
    reporter.solver_step_started("step A")

    # --- act --------------------------
    snapshot = reporter._build_snapshot(_stub_progress(), state, ignore_infeasible_diversity=False)
    reporter.show_update(replace(snapshot, debug_info="materialized"), get_debug_info=lambda: "callable")

    # --- assert -----------------------
    assert "materialized" in capsys.readouterr().out


def test_milestone_is_a_no_op_by_default():
    """Only renderers that can set a row apart override show_milestone; the base renders nothing."""
    # --- arrange ----------------------
    reporter = _RecordingProgressReporter()
    state = _stub_state()
    reporter.solver_step_started("step A")
    snapshot = reporter._build_snapshot(_stub_progress(), state, ignore_infeasible_diversity=False)

    # --- act --------------------------
    ProgressReporter.show_milestone(reporter, snapshot)

    # --- assert -----------------------
    assert [call[0] for call in reporter.calls] == ["started"]


def test_tabular_hash_column_blank_without_selection_or_hash(capsys):
    """A snapshot with neither the selection nor a precomputed hash renders an empty hash column."""
    # --- arrange ----------------------
    reporter = TabularProgressReporter()
    state = _stub_state()
    reporter.solver_step_started("step A")
    snapshot = reporter._build_snapshot(_stub_progress(), state, ignore_infeasible_diversity=False)

    # --- act --------------------------
    reporter.show_update(replace(snapshot, selection=None, selection_hash=None))

    # --- assert -----------------------
    row = capsys.readouterr().out.splitlines()[-1]
    assert row.rstrip().endswith("|")
    assert all(char in "| " for char in row.split("|")[-2])  # hash column is blank
