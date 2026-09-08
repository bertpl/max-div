from pathlib import Path

import pytest

from benchmarks.figures.style import TOOL_COLORS, UNLISTED_TOOL_COLOR, save_webp, tool_color, tool_key
from benchmarks.solver_scaling.configs import CONFIGS


def test_every_scaling_tool_has_a_color():
    """Each tool with a scaling configuration has its own palette entry, not the fallback."""
    # --- act / assert -----------------
    assert all(c.tool in TOOL_COLORS for c in CONFIGS)


def test_unlisted_tool_gets_the_fallback_color():
    """A tool the palette does not list is drawn in the neutral fallback tone."""
    # --- act / assert -----------------
    assert tool_color("greedy") == UNLISTED_TOOL_COLOR


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("max-div[DEFAULT]", "max-div"),
        ("RDKit[MaxMinPicker]", "rdkit"),
        ("apricot[facility-location]", "apricot-select"),
        ("code-FDM[FairFlow]", "code-fdm"),
        ("random", "random"),
    ],
)
def test_tool_key_maps_record_labels_to_palette_keys(label: str, expected: str):
    """A record's `tool[variant]` label resolves to the key `TOOL_COLORS` uses for that tool."""
    # --- act / assert -----------------
    assert tool_key(label) == expected
    assert tool_color(expected) == TOOL_COLORS[expected]


def test_save_webp_writes_a_webp_file(tmp_path: Path):
    """The shared writer produces a webp file and creates missing parent folders."""
    # --- arrange ----------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "nested" / "figure.webp"

    # --- act --------------------------
    save_webp(fig, path)

    # --- assert -----------------------
    assert path.read_bytes()[8:12] == b"WEBP"


def test_anytime_chart_draws_reference_lines_and_markers(tmp_path: Path):
    """A chart with a reference line and a marker is written; both take a legend entry."""
    # --- arrange ----------------------
    from benchmarks.common.records import RunRecord
    from benchmarks.figures import ReferenceLine, ReferenceMarker, plot_anytime_curve

    records = [
        RunRecord("max-div[DEFAULT]", "U1", 20, 20, 2, "MIN_SEPARATION", 0, f"time:{b}s", b, None, {"MIN_SEPARATION": q})
        for b, q in ((0.001, 0.5), (1.0, 0.9))
    ]
    path = tmp_path / "chart.webp"

    # --- act --------------------------
    plot_anytime_curve(
        records,
        "MIN_SEPARATION",
        path,
        reference_lines=(ReferenceLine(1.0, "certified optimum"),),
        reference_markers=(ReferenceMarker(0.3, 1.0, "SCIP proof"),),
    )

    # --- assert -----------------------
    assert path.read_bytes()[8:12] == b"WEBP"


def test_budget_series_stats_follow_the_budget_order():
    """The curve is aggregated per budget in increasing budget order, whatever order the records come in."""
    # --- arrange ----------------------
    from benchmarks.common.records import RunRecord
    from benchmarks.figures.anytime import _budget_series_stats

    records = [
        RunRecord("max-div[DEFAULT]", "U1", 20, 20, 2, "MIN_SEPARATION", 0, f"time:{b}s", t, None, {"MIN_SEPARATION": q})
        for b, t, q in ((0.1, 0.1, 0.6), (0.001, 0.03, 0.5), (0.05, 0.055, 0.55))
    ]

    # --- act --------------------------
    t_mean, q_mean, _q_min, _q_max = _budget_series_stats(records, "MIN_SEPARATION")

    # --- assert -----------------------
    assert q_mean == [0.5, 0.55, 0.6]
    assert t_mean == [0.03, 0.055, 0.1]


def test_charted_budgets_skip_budgets_below_the_previous_charted_measured_time():
    """Budgets that end at the set-up cost collapse to their first member; a budget past the previous measured time is charted again."""
    # --- arrange ----------------------
    from benchmarks.figures.anytime import _charted_budgets

    t_mean_by_budget = {
        "time:0.001s": 0.03,
        "time:0.002s": 0.03,
        "time:0.02s": 0.031,
        "time:0.05s": 0.055,
        "time:0.1s": 0.1,
        "iterations:10": 0.001,
    }

    # --- act / assert -----------------
    assert _charted_budgets(t_mean_by_budget) == ["time:0.001s", "time:0.05s", "time:0.1s", "iterations:10"]
