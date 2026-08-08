from pathlib import Path

from benchmarks.common.records import RunRecord, save_records
from benchmarks.tier1.report import build_incumbent_table, build_maxmin_gap_table, build_scaling_table, main


def _record(problem: str, size: int, budget: str, quality: dict[str, float]) -> RunRecord:
    """Minimal max-div ladder record with only the fields the report helpers read."""
    return RunRecord(
        tool="max-div[SMART]",
        problem=problem,
        size=size,
        n=100 * size,
        k=10 * size,
        diversity_metric=next(iter(quality)),
        seed=0,
        budget=budget,
        measured_sec=1.0,
        n_iterations=None,
        quality=quality,
    )


def test_build_maxmin_gap_table():
    # --- arrange -----------------------------------------
    exact_rows = [
        {"problem": "U1", "size": 1, "n": 100, "k": 10, "m": 0, "optimum": 0.1, "measured_sec": 0.31},
    ]
    records = [_record("U1", 1, f"time:{b}s", {"MIN_SEPARATION": 0.09}) for b in (0.016, 0.128, 1.024, 16.384)]

    # --- act ---------------------------------------------
    table = build_maxmin_gap_table(exact_rows, records)

    # --- assert ------------------------------------------
    assert "| U1 | 100 | 10 | 0 | 0.1000 | 0.31 s |" in table
    assert table.count("10.0%") == 4  # (0.1 - 0.09) / 0.1 at every quoted budget
    header, separator = table.splitlines()[0], table.splitlines()[1]
    assert header.count("|") == separator.count("|")  # else it renders as text, not a table


def test_build_scaling_table_marks_timeouts_and_gaps():
    # --- arrange -----------------------------------------
    rows = [
        {"backend": "SCIP (1 thread)", "n": 40, "k": 4, "measured_sec": 39.6, "proven": True},
        {"backend": "SCIP (1 thread)", "n": 50, "k": 5, "measured_sec": 900.0, "proven": False},
        {"backend": "CP-SAT (8 workers)", "n": 40, "k": 4, "measured_sec": 0.8, "proven": True},
        {"backend": "CP-SAT (8 workers)", "n": 50, "k": 5, "measured_sec": 2.1, "proven": True},
    ]

    # --- act ---------------------------------------------
    table = build_scaling_table(rows)

    # --- assert ------------------------------------------
    assert "| 40 | 4 | 39.6 s | 0.8 s |" in table
    assert "| 50 | 5 | **timeout** | 2.1 s |" in table


def test_build_incumbent_table():
    # --- arrange -----------------------------------------
    panel_rows = [
        {
            "problem": "U3",
            "size": 1,
            "n": 100,
            "k": 10,
            "m": 0,
            "cap_sec": 10800.0,
            "objective_value": 1.0,
            "objective_bound": 4.5,
        }
    ]
    records = [
        _record("U3", 1, "time:1.024s", {"GEOMEAN_SEPARATION": 0.98}),
        _record("U3", 1, "time:1.024s", {"GEOMEAN_SEPARATION": 0.99}),
    ]

    # --- act ---------------------------------------------
    table = build_incumbent_table(panel_rows, records)

    # --- assert ------------------------------------------
    assert "| U3 | 100 | 10 | 0 | 10800 s | 1.0000 | 350% | 0.9900 |" in table


def test_main_reads_tracked_exact_references(tmp_path: Path):
    # --- arrange -----------------------------------------
    # Fresh max-div records for every cell the tracked exact references cover.
    maxmin = [
        _record(problem, size, f"time:{b}s", {"MIN_SEPARATION": 0.05})
        for problem in ("U1", "C1")
        for size in (1, 2, 3)
        for b in (0.016, 0.128, 1.024, 16.384)
    ]
    incumbent = [_record(problem, 1, "time:1.024s", {"GEOMEAN_SEPARATION": 0.5}) for problem in ("U3", "C4")]
    records_dir = tmp_path / "records"
    results_dir = tmp_path / "results"
    save_records(maxmin, records_dir / "maxmin_records.jsonl")
    save_records(incumbent, records_dir / "incumbent_records.jsonl")

    # --- act ---------------------------------------------
    main(records_dir=records_dir, results_dir=results_dir)

    # --- assert ------------------------------------------
    gap_table = (results_dir / "tier1_maxmin_gap.md").read_text()
    data_rows = [line for line in gap_table.splitlines() if line.startswith(("| U1 |", "| C1 |"))]
    assert len(data_rows) == 6  # every (problem, size) cell of the tracked exact references
    scaling_table = (results_dir / "tier1_scaling.md").read_text()
    assert "SCIP (1 thread)" in scaling_table
    incumbent_table = (results_dir / "tier1_incumbent_geomean.md").read_text()
    assert "| U3 |" in incumbent_table
    assert "| C4 |" in incumbent_table
