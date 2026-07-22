import pytest

from benchmarks.common.records import RunRecord
from benchmarks.tier2.report import (
    best_competitor_mean,
    build_margin_table,
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
