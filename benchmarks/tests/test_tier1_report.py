import json
from pathlib import Path

import pytest

from benchmarks.common.records import RunRecord, save_records
from benchmarks.tier1 import report
from max_div.metrics import DiversityMetric


def _record(problem: str, n: int, tool: str, budget: str, quality: float, metric: str = "MIN_SEPARATION") -> RunRecord:
    """Minimal max-div budget-series record with only the fields the report reads."""
    return RunRecord(
        tool=tool,
        problem=problem,
        size=n,
        n=n,
        k=n // 10,
        diversity_metric=metric,
        seed=0,
        budget=budget,
        measured_sec=1.0,
        n_iterations=None,
        quality={metric: quality},
    )


def _exact(problem: str, objective: str, solver: str, n: int, optimum: float, proven: bool = True) -> dict:
    return {
        "problem": problem,
        "objective": objective,
        "solver": solver,
        "n": n,
        "k": n // 10,
        "m": 0,
        "optimum": optimum,
        "proven_optimal": proven,
        "measured_sec": 2.5,
    }


def test_gap_table_quotes_both_series_at_both_budgets():
    # --- arrange -----------------------------------------
    exact = [_exact("U1", "MIN_SEPARATION", "scip", 20, 1.0), _exact("U1", "MIN_SEPARATION", "highs", 20, 1.0)]
    single, multi = "max-div[DEFAULT]", "max-div[DEFAULT, 12 workers]"
    records = [
        _record("U1", 20, single, "time:1.0s", 0.9),
        _record("U1", 20, single, "time:60.0s", 0.95),
        _record("U1", 20, multi, "time:1.0s", 0.92),
        _record("U1", 20, multi, "time:60.0s", 1.0),
    ]

    # --- act ---------------------------------------------
    table = report.build_gap_table(exact, records, DiversityMetric.MIN_SEPARATION)

    # --- assert ------------------------------------------
    row = next(line for line in table.splitlines() if line.startswith("| U1 |"))
    assert "SCIP (PySCIPOpt) (2.5 s), HiGHS (2.5 s)" in row
    assert row.endswith("| 10.0% | 5.0% | 8.0% | 0.0% |")


def test_certification_table_states_where_each_column_stopped():
    # --- arrange -----------------------------------------
    exact = [
        _exact("U1", "MIN_SEPARATION", "scip", 20, 1.0),
        _exact("U1", "MIN_SEPARATION", "scip", 50, 1.0, proven=False),
        _exact("C1", "MEAN_SEPARATION", "ortools-cpsat", 20, 1.0),
    ]

    # --- act ---------------------------------------------
    table = report.build_certification_table(exact)

    # --- assert ------------------------------------------
    assert "| SCIP (PySCIPOpt) | U1 | MIN_SEPARATION | **20** | n=50 (2 s, not certified) |" in table
    assert "| OR-Tools CP-SAT | C1 | MEAN_SEPARATION | **20** | grid exhausted |" in table


def test_gap_pct_is_none_without_a_value():
    # --- act / assert ------------------------------------
    assert report.gap_pct(None, 1.0) is None
    assert report.gap_pct(0.9, 1.0) == pytest.approx(10.0)


def test_main_emits_tables_charts_and_gallery_snippets(tmp_path: Path):
    """The report writes one chart per certified cell, a full-width list for min separation, and galleries otherwise."""
    # --- arrange -----------------------------------------
    data_dir, records_dir, docs_dir = tmp_path / "data", tmp_path / "records", tmp_path / "docs"
    data_dir.mkdir()
    (data_dir / report.EXACT_MAXMIN_FILE).write_text(json.dumps([_exact("U1", "MIN_SEPARATION", "scip", 20, 1.0)]))
    (data_dir / report.EXACT_NN_FILE).write_text(json.dumps([_exact("U1", "GEOMEAN_SEPARATION", "ortools-cpsat", 20, 1.0)]))
    for metric in ("MIN_SEPARATION", "GEOMEAN_SEPARATION"):
        records = [_record("U1", 20, "max-div[DEFAULT]", f"time:{b}s", 0.9, metric) for b in (1.0, 60.0)]
        save_records(records, records_dir / f"maxdiv_{metric.lower()}.jsonl")

    # --- act ---------------------------------------------
    report.main(records_dir=records_dir, docs_dir=docs_dir, data_dir=data_dir)

    # --- assert ------------------------------------------
    assert (docs_dir / "images" / "tier1_U1_20_min_separation.webp").exists()
    assert (docs_dir / "images" / "tier1_U1_20_geomean_separation.webp").exists()
    assert "![tier1_U1_20_min_separation]" in (docs_dir / "results" / "tier1_charts_min_separation.md").read_text()
    assert 'width="32%"' in (docs_dir / "results" / "tier1_gallery_geomean_separation.md").read_text()
    assert (docs_dir / "results" / "tier1_gallery_mean_separation.md").read_text() == "\n"
    assert "| U1 | 20 |" in (docs_dir / "results" / "tier1_gap_min_separation.md").read_text()
