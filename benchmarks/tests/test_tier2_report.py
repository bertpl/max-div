from pathlib import Path

import pytest

from benchmarks.common.records import RunRecord, save_records
from benchmarks.tier2.report import (
    TABLE_METRICS,
    best_competitor_mean,
    build_margin_table,
    main,
    margin_pct,
    maxdiv_mean_at_budget,
    records_for_figure,
)


def _record(tool: str, budget: str, quality: float, diversity_metric: str = "MIN_SEPARATION") -> RunRecord:
    """Minimal record with only the fields the report helpers read."""
    return RunRecord(
        tool=tool,
        problem="U1",
        size=2,
        n=200,
        k=20,
        diversity_metric=diversity_metric,
        seed=0,
        budget=budget,
        measured_sec=1.0,
        n_iterations=None,
        quality={"MIN_SEPARATION": quality},
    )


def test_records_for_figure_filters_maxdiv_to_optimizing_run():
    # --- arrange -----------------------------------------
    records = [
        _record("max-div[SMART]", "time:1.024s", 1.0, diversity_metric="MIN_SEPARATION"),
        _record("max-div[SMART]", "time:1.024s", 2.0, diversity_metric="GEOMEAN_SEPARATION"),
        _record("fpsample[FPS]", "single-shot", 3.0, diversity_metric="GEOMEAN_SEPARATION"),
    ]

    # --- act ---------------------------------------------
    selected = records_for_figure(records, "MIN_SEPARATION")

    # --- assert ------------------------------------------
    assert len(selected) == 2
    assert {r.tool for r in selected} == {"max-div[SMART]", "fpsample[FPS]"}


def test_margin_pct_positive_when_maxdiv_wins():
    # --- arrange -----------------------------------------
    records = [
        _record("max-div[SMART]", "time:1.024s", 1.1),
        _record("fpsample[FPS]", "single-shot", 1.0),
        _record("random", "single-shot", 0.5),
    ]

    # --- act / assert ------------------------------------
    assert maxdiv_mean_at_budget(records, "MIN_SEPARATION", 1.024) == pytest.approx(1.1)
    assert best_competitor_mean(records, "MIN_SEPARATION") == ("fpsample[FPS]", pytest.approx(1.0))
    assert margin_pct(records, "MIN_SEPARATION", 1.024) == pytest.approx(10.0)


def test_margin_pct_none_when_budget_missing():
    # --- arrange -----------------------------------------
    records = [_record("fpsample[FPS]", "single-shot", 1.0)]

    # --- act / assert ------------------------------------
    assert margin_pct(records, "MIN_SEPARATION", 1.024) is None


def test_build_margin_table_shape():
    # --- arrange -----------------------------------------
    records = [
        _record("max-div[SMART]", "time:1.024s", 1.1),
        _record("max-div[SMART]", "time:16.384s", 1.2),
        _record("fpsample[FPS]", "single-shot", 1.0),
    ]

    # --- act ---------------------------------------------
    table = build_margin_table(records, "MIN_SEPARATION", problems=["U1"], sizes=[2])

    # --- assert ------------------------------------------
    assert "| n | U1 |" in table
    assert "| 200 | +10.0% / +20.0% |" in table


def _fresh_maxdiv_record(problem: str, budget: str, diversity_metric: str) -> RunRecord:
    """A max-div budget-series record as a re-measurement would produce it, quality 1.0 under every metric."""
    return RunRecord(
        tool="max-div[DEFAULT]",
        problem=problem,
        size=2,
        n=200,
        k=20,
        diversity_metric=diversity_metric,
        seed=0,
        budget=budget,
        measured_sec=1.0,
        n_iterations=100,
        quality=dict.fromkeys(TABLE_METRICS, 1.0),
    )


def test_main_merges_tracked_third_party_with_fresh_maxdiv(tmp_path: Path):
    # --- arrange -----------------------------------------
    records_dir = tmp_path / "records"
    docs_dir = tmp_path / "docs"
    budgets = ("time:1.024s", "time:16.384s")
    unconstrained = [_fresh_maxdiv_record("U1", b, m) for m in TABLE_METRICS for b in budgets]
    constrained = [_fresh_maxdiv_record("C1", b, "MIN_SEPARATION") for b in budgets]
    save_records(unconstrained, records_dir / "maxdiv_unconstrained.jsonl")
    save_records(constrained, records_dir / "maxdiv_constrained.jsonl")

    # --- act ---------------------------------------------
    main(records_dir=records_dir, docs_dir=docs_dir)

    # --- assert ------------------------------------------
    table = (docs_dir / "results" / "tier2_margins_min_separation.md").read_text()
    row = next(line for line in table.splitlines() if line.startswith("| 200 |"))
    cells = [c.strip() for c in row.split("|")[2:-1]]
    assert "%" in cells[0]  # U1: fresh max-div vs. tracked competitors -> a margin
    assert cells[1] == "-"  # U2: no fresh max-div records planted
    constrained_table = (docs_dir / "results" / "tier2_margins_constrained.md").read_text()
    assert "%" in constrained_table
    assert (docs_dir / "images" / "tier2_U1_2_min_separation.webp").exists()
    assert (docs_dir / "images" / "tier2_C1_2_min_separation.webp").exists()
