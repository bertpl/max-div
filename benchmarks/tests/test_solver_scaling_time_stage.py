from benchmarks.solver_scaling import time_stage
from benchmarks.solver_scaling.configs import resolve
from benchmarks.solver_scaling.grid import DEFAULT_SEED, WARMUP_BUDGET_SEC
from benchmarks.solver_scaling.records import ScalingRunRecord, load_scaling_records


def _record(n, seed=DEFAULT_SEED, *, completed=True, reason=None, measured_sec=1.0):
    """Build one time-stage run record; overrides drive the pass/fail and crash cases."""
    return ScalingRunRecord("rdkit", "default", n, n // 10, seed, 60.0, completed, reason, measured_sec, 1000, 0.2)


def test_passes_time_requires_completion_within_the_budget():
    # --- act / assert -----------------
    assert time_stage.passes_time(_record(100, 0, measured_sec=59.0), budget_sec=60.0)
    assert not time_stage.passes_time(_record(100, 0, measured_sec=61.0), budget_sec=60.0)
    assert not time_stage.passes_time(_record(100, 0, completed=False), budget_sec=60.0)


def test_ascend_stops_at_the_first_failing_size(monkeypatch, tmp_path):
    # --- arrange ----------------------
    calls = []

    def fake_run(tool, config, n, k, seed, budget_sec):
        calls.append(n)
        return _record(n, seed, completed=(n <= 100))

    monkeypatch.setattr(time_stage, "run_measurement", fake_run)

    # --- act --------------------------
    limit = time_stage._ascend(resolve("rdkit", "default"), {}, tmp_path / "runs.jsonl", 60.0)

    # --- assert -----------------------
    assert limit == 100
    assert max(calls) == 200  # ascends to the first failing size, then stops


def test_ascend_reuses_already_recorded_runs(monkeypatch, tmp_path):
    # --- arrange ----------------------
    calls = []

    def fake_run(tool, config, n, k, seed, budget_sec):
        calls.append(n)
        return _record(n, seed, completed=(n <= 50))

    monkeypatch.setattr(time_stage, "run_measurement", fake_run)
    done = {("rdkit", "default", 20, DEFAULT_SEED): _record(20, DEFAULT_SEED)}

    # --- act --------------------------
    limit = time_stage._ascend(resolve("rdkit", "default"), done, tmp_path / "runs.jsonl", 60.0)

    # --- assert -----------------------
    assert limit == 50
    assert 20 not in calls  # the recorded size was not re-run


def test_a_fresh_config_gets_one_discarded_warmup_run(monkeypatch, tmp_path):
    """The warm-up run precedes the sweep, uses a short budget, and is not recorded."""
    # --- arrange ----------------------
    calls = []

    def fake_run(tool, config, n, k, seed, budget_sec):
        calls.append((n, budget_sec))
        return _record(n, completed=(n <= 50))

    monkeypatch.setattr(time_stage, "run_measurement", fake_run)
    data_path = tmp_path / "runs.jsonl"

    # --- act --------------------------
    time_stage._ascend(resolve("rdkit", "default"), {}, data_path, 60.0)

    # --- assert -----------------------
    assert calls[0] == (20, WARMUP_BUDGET_SEC)
    assert calls[1] == (20, 60.0)
    recorded = load_scaling_records(data_path)
    assert all(r.budget_sec == 60.0 for r in recorded)  # the warm-up run was discarded


def test_ascend_skips_a_crash_before_any_pass_but_a_timeout_stops(monkeypatch, tmp_path):
    """A non-resource crash with no size passed yet is skipped; a later timeout still ends the sweep."""
    # --- arrange ----------------------
    def fake_run(tool, config, n, k, seed, budget_sec):
        if n < 100:
            return _record(n, completed=False, reason="RuntimeError: degenerate")  # crash, no pass yet
        if n > 1000:
            return _record(n, completed=False, reason="timeout")  # a real, monotonic time limit
        return _record(n)  # passes at 100, 200, 500, 1000

    monkeypatch.setattr(time_stage, "run_measurement", fake_run)

    # --- act --------------------------
    limit = time_stage._ascend(resolve("rdkit", "default"), {}, tmp_path / "runs.jsonl", 60.0)

    # --- assert -----------------------
    assert limit == 1000  # the n<100 crashes were skipped; the sweep stopped at the n=2000 timeout
