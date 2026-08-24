import pytest

from benchmarks.solver_scaling import extended_stage
from benchmarks.solver_scaling.configs import resolve
from benchmarks.solver_scaling.grid import DEFAULT_SEED, EXTENDED_BUDGET_SEC
from benchmarks.solver_scaling.records import ScalingRunRecord, append_scaling_record


def _record(n, *, completed=True, reason=None, measured_sec=1.0, min_separation=0.2, budget_sec=EXTENDED_BUDGET_SEC):
    """Build one extended-stage run record; overrides drive the stopping and best-known cases."""
    return ScalingRunRecord(
        "rdkit", "default", n, n // 10, DEFAULT_SEED, budget_sec, completed, reason, measured_sec, 1000, min_separation
    )


def test_ascend_keeps_going_past_a_late_completion_but_stops_on_a_kill(monkeypatch, tmp_path):
    """A completed run over the extended budget continues the series; a timeout kill ends it."""
    # --- arrange ----------------------
    def fake_run(tool, config, n, k, seed, budget_sec):
        if n >= 500:
            return _record(n, completed=False, reason="timeout")
        # completes 10 s past the budget: late, but a solution was produced
        return _record(n, measured_sec=budget_sec + 10.0)

    monkeypatch.setattr(extended_stage, "run_measurement", fake_run)

    # --- act --------------------------
    largest = extended_stage._ascend(resolve("rdkit", "default"), {}, tmp_path / "runs.jsonl", 5000, 900.0)

    # --- assert -----------------------
    assert largest == 200  # every late completion counted; the n=500 kill ended the series


def test_ascend_stops_on_a_memory_kill_and_on_a_crash_after_a_success(monkeypatch, tmp_path):
    # --- arrange ----------------------
    def fake_run_memory(tool, config, n, k, seed, budget_sec):
        return _record(n, completed=(n <= 50), reason=None if n <= 50 else "memory")

    def fake_run_crash(tool, config, n, k, seed, budget_sec):
        return _record(n, completed=(n <= 50), reason=None if n <= 50 else "ValueError: boom")

    config = resolve("rdkit", "default")

    # --- act / assert -----------------
    monkeypatch.setattr(extended_stage, "run_measurement", fake_run_memory)
    assert extended_stage._ascend(config, {}, tmp_path / "a.jsonl", 5000, 900.0) == 50
    monkeypatch.setattr(extended_stage, "run_measurement", fake_run_crash)
    assert extended_stage._ascend(config, {}, tmp_path / "b.jsonl", 5000, 900.0) == 50


def test_ascend_skips_a_crash_before_any_completed_size(monkeypatch, tmp_path):
    """A non-resource crash with nothing completed yet is a small-size degeneracy, not a limit."""
    # --- arrange ----------------------
    def fake_run(tool, config, n, k, seed, budget_sec):
        if n < 100:
            return _record(n, completed=False, reason="RuntimeError: degenerate")
        return _record(n, completed=(n <= 200))

    monkeypatch.setattr(extended_stage, "run_measurement", fake_run)

    # --- act --------------------------
    largest = extended_stage._ascend(resolve("rdkit", "default"), {}, tmp_path / "runs.jsonl", 5000, 900.0)

    # --- assert -----------------------
    assert largest == 200


def test_ascend_walks_only_up_to_the_size_bound(monkeypatch, tmp_path):
    # --- arrange ----------------------
    calls = []

    def fake_run(tool, config, n, k, seed, budget_sec):
        calls.append(n)
        return _record(n)

    monkeypatch.setattr(extended_stage, "run_measurement", fake_run)

    # --- act --------------------------
    largest = extended_stage._ascend(resolve("rdkit", "default"), {}, tmp_path / "runs.jsonl", 500, 900.0)

    # --- assert -----------------------
    assert largest == 500
    assert max(calls) == 500  # nothing beyond the bound was attempted


def test_size_bound_comes_from_the_time_stage_passes(tmp_path):
    """The bound is the largest size any configuration passed within the time budget."""
    # --- arrange ----------------------
    path = tmp_path / "time_stage.jsonl"
    append_scaling_record(_record(500, measured_sec=59.0, budget_sec=60.0), path)
    append_scaling_record(_record(1000, measured_sec=61.0, budget_sec=60.0), path)  # over budget: no pass

    # --- act / assert -----------------
    assert extended_stage.size_bound_from_time_stage(path) == 500


def test_best_known_by_size_takes_the_best_completed_run_per_size():
    # --- arrange ----------------------
    records = [
        _record(100, min_separation=0.3),
        ScalingRunRecord("max-div", "lean", 100, 10, DEFAULT_SEED, 900.0, True, None, 800.0, 1000, 0.5),
        _record(100, completed=False, min_separation=0.9),  # not completed: never enters the pool
        _record(200, min_separation=None),  # no quality recorded: never enters the pool
    ]

    # --- act --------------------------
    best = extended_stage.best_known_by_size(records)

    # --- assert -----------------------
    assert list(best) == [100]
    assert best[100].tool == "max-div"
    assert best[100].min_separation == pytest.approx(0.5)
