"""Guards for the subprocess runner: the completed path, both kill paths, and records."""

from pathlib import Path

from benchmarks.tool_scaling.configs import Mode
from benchmarks.tool_scaling.records import ScalingRunRecord, append_scaling_record, load_scaling_records
from benchmarks.tool_scaling.runner import run_measurement


def test_a_completed_run_reports_time_memory_and_quality() -> None:
    """The full round-trip: subprocess, child-reported timing, peak memory, min separation."""
    # --- act --------------------------
    record = run_measurement("max-div", Mode.FASTEST_VALID, n=100, k=10, seed=0, budget_sec=30.0)

    # --- assert -----------------------
    assert record.completed, record.reason
    assert record.measured_sec is not None and record.measured_sec < 30.0
    assert record.peak_rss_bytes and record.peak_rss_bytes > 0
    assert record.min_separation and record.min_separation > 0.0


def test_a_run_past_its_deadline_is_killed_and_recorded_as_timeout(monkeypatch) -> None:
    """The parent's deadline kill lands as a `timeout` record, not an exception."""
    # --- arrange ----------------------
    monkeypatch.setattr("benchmarks.tool_scaling.runner.SETUP_GRACE_SEC", 20.0)

    # --- act --------------------------
    record = run_measurement("_test_sleep", Mode.FASTEST_VALID, n=100, k=10, seed=0, budget_sec=1.0)

    # --- assert -----------------------
    assert not record.completed
    assert record.reason == "timeout"


def test_a_child_error_reaches_the_parent_as_a_failed_record() -> None:
    """An unknown tool raises inside the child; the parent must record it, not crash."""
    # --- act --------------------------
    record = run_measurement("no-such-tool", Mode.FASTEST_VALID, n=100, k=10, seed=0, budget_sec=5.0)

    # --- assert -----------------------
    assert not record.completed
    assert record.reason and "KeyError" in record.reason


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    """Appending and loading preserves every field of a record."""
    # --- arrange ----------------------
    record = ScalingRunRecord(
        tool="max-div",
        mode="fastest_valid",
        n=100,
        k=10,
        seed=0,
        budget_sec=60.0,
        completed=True,
        reason=None,
        measured_sec=0.5,
        peak_rss_bytes=123456,
        min_separation=0.1,
    )
    path = tmp_path / "records.jsonl"

    # --- act --------------------------
    append_scaling_record(record, path)
    append_scaling_record(record, path)

    # --- assert -----------------------
    assert load_scaling_records(path) == [record, record]
