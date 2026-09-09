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
        _record("fpsample[vanilla]", "single-shot", 1.0, measured_sec=0.01),
        _record("RDKit MaxMinPicker[default]", "single-shot", 1.2, measured_sec=0.5),
        _record("max-div[DEFAULT]", "time:1.0s", 9.0),
    ]

    # --- act / assert -----------------
    assert report.best_entrant(records) == ("RDKit MaxMinPicker[default]", pytest.approx(1.2), pytest.approx(0.5))


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


def test_size_table_orders_every_tool_by_quality_and_names_the_overtake_budgets():
    """One table: max-div's quoted budgets and the entrants, best first, faster first on a tie, then the overtake sentence."""
    # --- arrange ----------------------
    records = [
        _record("fpsample[vanilla]", "single-shot", 1.0, measured_sec=0.02),
        _record("skmatter[default]", "single-shot", 0.99996, measured_sec=0.01),  # prints as 1.0000: a tie
        _record("max-div[DEFAULT]", "time:1.0s", 0.9, measured_sec=1.0),
        _record("max-div[DEFAULT]", "time:60.0s", 1.2, measured_sec=60.0),
        _record("max-div[DEFAULT, 12 workers]", "time:1.0s", 1.05, measured_sec=1.35),
        _record("max-div[DEFAULT, 12 workers]", "time:60.0s", 1.3, measured_sec=60.4),
    ]

    # --- act --------------------------
    table = report.build_size_table(records, 200)

    # --- assert -----------------------
    assert table.splitlines()[2:8] == [
        "| max-div[DEFAULT, 12 workers] @ 60 s | 1.3000 | 60.4 s |",
        "| max-div[DEFAULT] @ 60 s | 1.2000 | 60 s |",
        "| max-div[DEFAULT, 12 workers] @ 1 s | 1.0500 | 1.35 s |",
        "| skmatter[default] | 1.0000 | 0.01 s |",
        "| fpsample[vanilla] | 1.0000 | 0.02 s |",
        "| max-div[DEFAULT] @ 1 s | 0.9000 | 1 s |",
    ]
    assert table.endswith("at a budget of 60 s with one worker and at a budget of 1 s with 12 workers.\n")


def test_main_emits_chart_per_size_with_tables(tmp_path: Path):
    """The report writes one chart and one table per size."""
    # --- arrange ----------------------
    data_dir, records_dir, docs_dir = tmp_path / "data", tmp_path / "records", tmp_path / "docs"
    save_records([_record("fpsample[vanilla]", "single-shot", 1.0)], data_dir / report.ENTRANT_FILE)
    save_records(
        [_record("max-div[DEFAULT]", f"time:{b}s", 0.9) for b in (1.0, 60.0)], records_dir / report.MAXDIV_FILE
    )

    # --- act --------------------------
    report.main(records_dir=records_dir, docs_dir=docs_dir, data_dir=data_dir)

    # --- assert -----------------------
    assert (docs_dir / "images" / "tier2_U1_200_min_separation.webp").exists()
    assert "| fpsample[vanilla] |" in (docs_dir / "results" / "tier2_size_200.md").read_text()


def test_render_charts_leaves_the_random_baseline_off_the_chart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The random record never reaches the plotter, while the entrants and max-div do."""
    # --- arrange ----------------------
    plotted: list[list[RunRecord]] = []
    monkeypatch.setattr(report, "plot_anytime_curve", lambda records, **kwargs: plotted.append(records))
    records = [
        _record("random", "single-shot", 0.01),
        _record("fpsample[vanilla]", "single-shot", 1.0),
        _record("max-div[DEFAULT]", "time:1.0s", 1.2),
    ]

    # --- act --------------------------
    report.render_charts(records, [200], tmp_path)

    # --- assert -----------------------
    assert [r.tool for r in plotted[0]] == ["fpsample[vanilla]", "max-div[DEFAULT]"]
