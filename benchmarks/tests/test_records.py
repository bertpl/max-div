from pathlib import Path

from benchmarks.common import RunRecord, load_records, save_records


def test_records_round_trip(tmp_path: Path):
    # --- arrange -----------------------------------------
    records = [
        RunRecord(
            tool="max-div[SMART]",
            problem="U1",
            size=2,
            n=200,
            k=20,
            diversity_metric="GEOMEAN_SEPARATION",
            seed=1,
            budget="time:0.004s",
            measured_sec=0.0051,
            n_iterations=123,
            quality={"MIN_SEPARATION": 0.5, "GEOMEAN_SEPARATION": 0.9},
            n_constraints=2,
            n_constraints_satisfied=2,
        ),
        RunRecord(
            tool="fpsample[FPS]",
            problem="U1",
            size=2,
            n=200,
            k=20,
            diversity_metric="GEOMEAN_SEPARATION",
            seed=1,
            budget="single-shot",
            measured_sec=0.0002,
            n_iterations=None,
            quality={"MIN_SEPARATION": 0.4},
            proven_optimal=None,
        ),
    ]

    # --- act ---------------------------------------------
    path = tmp_path / "sub" / "records.jsonl"
    save_records(records, path)
    loaded = load_records(path)

    # --- assert ------------------------------------------
    assert loaded == records
