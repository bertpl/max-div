from benchmarks.solver_scaling.records import (
    ScalingRunRecord,
    append_scaling_record,
    load_scaling_records,
    save_scaling_records,
)


def _record(**overrides) -> ScalingRunRecord:
    fields = {
        "tool": "max-div",
        "config": "lean",
        "n": 100,
        "k": 10,
        "seed": 0,
        "budget_sec": 60.0,
        "completed": True,
        "reason": None,
        "measured_sec": 1.5,
        "peak_memory_bytes": 123456,
        "min_separation": 0.42,
    }
    fields.update(overrides)
    return ScalingRunRecord(**fields)


def test_save_then_load_round_trips_records(tmp_path):
    # --- arrange ----------------------
    records = [_record(n=100), _record(n=200, completed=False, reason="timeout", min_separation=None)]
    path = tmp_path / "sub" / "runs.jsonl"

    # --- act --------------------------
    save_scaling_records(records, path)

    # --- assert -----------------------
    assert load_scaling_records(path) == records


def test_append_adds_one_line_at_a_time(tmp_path):
    # --- arrange ----------------------
    path = tmp_path / "runs.jsonl"

    # --- act --------------------------
    append_scaling_record(_record(n=100), path)
    append_scaling_record(_record(n=200), path)

    # --- assert -----------------------
    loaded = load_scaling_records(path)
    assert [r.n for r in loaded] == [100, 200]
