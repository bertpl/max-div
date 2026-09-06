from pathlib import Path

import pytest

from benchmarks.common.records import RunRecord, save_records
from benchmarks.tier2 import report


def _record(tool: str, budget: str, quality: float, n: int = 200, measured_sec: float = 1.0) -> RunRecord:
    """Minimal record with only the fields the report reads."""
    return RunRecord(
        tool=tool,
        problem="U1",
        size=n,
        n=n,
        k=n // 10,
        diversity_metric="MIN_SEPARATION",
        seed=0,
        budget=budget,
        measured_sec=measured_sec,
        n_iterations=None,
        quality={"MIN_SEPARATION": quality},
    )


def test_best_entrant_ignores_the_random_baseline():
    """The random baseline is never the best entrant, however high its value."""
    # --- arrange ----------------------
    records = [
        _record("random", "single-shot", 5.0),
        _record("fpsample[FPS]", "single-shot", 1.0, measured_sec=0.01),
        _record("RDKit[MaxMinPicker]", "single-shot", 1.2, measured_sec=0.5),
        _record("max-div[DEFAULT]", "time:1.0s", 9.0),
    ]

    # --- act / assert -----------------
    assert report.best_entrant(records) == ("RDKit[MaxMinPicker]", pytest.approx(1.2), pytest.approx(0.5))


def test_overtake_budget_is_the_first_budget_whose_median_reaches_the_target():
    """The overtake budget is the first budget whose median reaches the target, None when none does."""
    # --- arrange ----------------------
    records = [
        _record("max-div[DEFAULT]", "time:0.001s", 0.5),
        _record("max-div[DEFAULT]", "time:1.0s", 1.1),
        _record("max-div[DEFAULT]", "time:60.0s", 1.3),
    ]

    # --- act / assert -----------------
    assert report.overtake_budget(records, "max-div[DEFAULT]", 1.0) == 1.0
    assert report.overtake_budget(records, "max-div[DEFAULT]", 2.0) is None


def test_summary_table_row():
    """One size's row carries the best entrant, both series' medians, and both overtake budgets."""
    # --- arrange ----------------------
    records = [
        _record("fpsample[FPS]", "single-shot", 1.0, measured_sec=0.02),
        _record("max-div[DEFAULT]", "time:1.0s", 0.9),
        _record("max-div[DEFAULT]", "time:60.0s", 1.2),
        _record("max-div[DEFAULT, 12 workers]", "time:1.0s", 1.05),
        _record("max-div[DEFAULT, 12 workers]", "time:60.0s", 1.3),
    ]

    # --- act --------------------------
    table = report.build_summary_table(records, [200])

    # --- assert -----------------------
    assert "| 200 | fpsample[FPS] | 1.0000 | 0.02 s | 0.9000 | 1.2000 | 1.0500 | 1.3000 | 60 s | 1 s |" in table


def test_main_emits_chart_per_size_with_tables(tmp_path: Path):
    """The report writes one chart per size, the chart list, and both tables."""
    # --- arrange ----------------------
    data_dir, records_dir, docs_dir = tmp_path / "data", tmp_path / "records", tmp_path / "docs"
    save_records([_record("fpsample[FPS]", "single-shot", 1.0)], data_dir / report.ENTRANT_FILE)
    save_records(
        [_record("max-div[DEFAULT]", f"time:{b}s", 0.9) for b in (1.0, 60.0)], records_dir / report.MAXDIV_FILE
    )

    # --- act --------------------------
    report.main(records_dir=records_dir, docs_dir=docs_dir, data_dir=data_dir)

    # --- assert -----------------------
    assert (docs_dir / "images" / "tier2_U1_200_min_separation.webp").exists()
    assert "| 200 |" in (docs_dir / "results" / "tier2_summary.md").read_text()
    assert "| fpsample[FPS] |" in (docs_dir / "results" / "tier2_entrants.md").read_text()
