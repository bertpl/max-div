from benchmarks.solver_scaling import time_stage
from benchmarks.solver_scaling.configs import resolve
from benchmarks.solver_scaling.grid import DEFAULT_SEED
from benchmarks.solver_scaling.records import ScalingRunRecord


def _record(n, seed=DEFAULT_SEED, *, completed=True, measured_sec=1.0):
    return ScalingRunRecord("rdkit", "default", n, n // 10, seed, 60.0, completed, None, measured_sec, 1000, 0.2)


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
