import pytest

from benchmarks.solver_scaling import memory_stage
from benchmarks.solver_scaling.configs import resolve
from benchmarks.solver_scaling.grid import DEFAULT_SEED, WARMUP_BUDGET_SEC
from benchmarks.solver_scaling.records import ScalingRunRecord, load_scaling_records


def _record(n, *, completed=True, reason=None, peak=None, settled=True, spawned=False):
    """Build one run record with a footprint following a 40 B/item baseline unless overridden."""
    peak = peak if peak is not None else int(1.6e8 + 40 * n)
    return ScalingRunRecord(
        "rdkit", "default", n, n // 10, DEFAULT_SEED, 60.0, completed, reason, 1.0, peak, 0.2, spawned, settled
    )


def test_sweep_stops_once_the_trust_conditions_hold(monkeypatch, tmp_path):
    """The sweep ends at the first size where the fit passes the trust conditions and publishes it."""
    # --- arrange ----------------------
    def fake_run(tool, config, n, k, seed, budget_sec):
        return _record(n)

    monkeypatch.setattr(memory_stage, "run_measurement", fake_run)

    # --- act --------------------------
    fit = memory_stage._sweep(resolve("rdkit", "default"), {}, tmp_path / "runs.jsonl")

    # --- assert -----------------------
    # 40 B/item growth doubles the ~160 MB baseline by n = 4M; the sweep stops at the next grid
    # size where both trust conditions hold and publishes the fitted crossing
    assert fit.coef is not None
    assert fit.r2 is not None and fit.r2 >= 0.95
    assert fit.max_n == 500_000_000  # (32 GB - c0) / 40 B, on the grid


def test_a_memory_kill_brackets_at_the_previous_size(monkeypatch, tmp_path):
    """A cap kill ends the sweep with the last under-cap size as the bracketed result."""
    # --- arrange ----------------------
    def fake_run(tool, config, n, k, seed, budget_sec):
        if n >= 200:
            return _record(n, completed=False, reason="memory", peak=None)
        return _record(n, peak=int(1.6e8))

    monkeypatch.setattr(memory_stage, "run_measurement", fake_run)

    # --- act --------------------------
    fit = memory_stage._sweep(resolve("rdkit", "default"), {}, tmp_path / "runs.jsonl")

    # --- assert -----------------------
    assert fit.max_n == 100
    assert fit.reason.startswith("bracketed: the memory cap")


def test_a_solver_failure_brackets_at_the_previous_size(monkeypatch, tmp_path):
    """A non-resource failure brackets like a cap kill, with the failure disclosed in the reason."""
    # --- arrange ----------------------
    def fake_run(tool, config, n, k, seed, budget_sec):
        if n >= 1000:
            return _record(n, completed=False, reason="ValueError: rank", peak=None)
        return _record(n, peak=int(1.6e8))

    monkeypatch.setattr(memory_stage, "run_measurement", fake_run)

    # --- act --------------------------
    fit = memory_stage._sweep(resolve("dppy", "default"), {}, tmp_path / "runs.jsonl")

    # --- assert -----------------------
    assert fit.max_n == 500
    assert "fails at the next size" in fit.reason


def test_a_config_observed_spawning_workers_is_not_measured(monkeypatch, tmp_path):
    """One observed worker-spawning run aborts the sweep with the disclosed exclusion."""
    # --- arrange ----------------------
    calls = []

    def fake_run(tool, config, n, k, seed, budget_sec):
        calls.append((n, budget_sec))
        return _record(n, spawned=True)

    monkeypatch.setattr(memory_stage, "run_measurement", fake_run)

    # --- act --------------------------
    fit = memory_stage._sweep(resolve("max-div", "optimal-lazy"), {}, tmp_path / "runs.jsonl")

    # --- assert -----------------------
    assert fit.max_n is None
    assert fit.reason == "not measured: spawns worker processes"
    assert calls == [(20, WARMUP_BUDGET_SEC), (20, 60.0)]  # the sweep aborted after one real run


def test_unsettled_footprints_still_feed_the_fit(monkeypatch, tmp_path):
    """The settled flag is diagnostic only: unsettled points feed the fit like any other."""
    # --- arrange ----------------------
    # every run reports an unsettled window (still growing at the kill), yet the fit must be
    # built from them and the sweep must terminate on the trust conditions
    def fake_run(tool, config, n, k, seed, budget_sec):
        return _record(n, completed=False, reason="timeout", settled=False)

    monkeypatch.setattr(memory_stage, "run_measurement", fake_run)

    # --- act --------------------------
    fit = memory_stage._sweep(resolve("rdkit", "default"), {}, tmp_path / "runs.jsonl")

    # --- assert -----------------------
    # same 40 B/item slope and crossing as the all-settled case, from unsettled points alone
    assert fit.coef is not None and fit.coef[1] == pytest.approx(40.0, rel=1e-6)
    assert fit.max_n == 500_000_000

    recorded = load_scaling_records(tmp_path / "runs.jsonl")
    assert all(r.budget_sec == 60.0 for r in recorded)  # the warm-up run was discarded
