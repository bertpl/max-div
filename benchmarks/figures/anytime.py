"""Plot anytime curves: quality vs. measured wall-clock, budget-series curves + single-shot dots + references."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.common.records import RunRecord, budget_sec, iteration_count

from .style import MARKER_SHAPES, save_webp, tool_color, tool_key, use_docs_style

# Heavier than the scaling charts' lines: one or two curves per chart, at a smaller figure size.
_LINE_WIDTH = 1.5
# Budget-series tools in legend order take these line styles in turn — the single-worker series
# solid, the multi-worker series dashed — so two max-div curves on one chart stay apart.
_SERIES_LINESTYLES = ("-", "--", "-.", ":")
_REFERENCE_COLOR = "#555555"
# The y-axis spans at least this fraction of the median plotted value: a series whose points all
# sit within float noise of a reference line would otherwise have the axis autoscaled to that noise.
_MIN_Y_SPAN_FRACTION = 0.01


@dataclass(frozen=True)
class ReferenceLine:
    """Describe a dotted horizontal line at a reference value, e.g. a certified optimum or the best entrant."""

    value: float
    label: str
    color: str = _REFERENCE_COLOR


@dataclass(frozen=True)
class ReferenceMarker:
    """Describe one marked point, e.g. an exact solver's certified optimum at its proof time."""

    x_sec: float
    value: float
    label: str
    color: str = _REFERENCE_COLOR
    marker: str = "X"


def plot_anytime_curve(
    records: list[RunRecord],
    metric_name: str,
    path: Path,
    title: str = "",
    reference_lines: tuple[ReferenceLine, ...] = (),
    reference_markers: tuple[ReferenceMarker, ...] = (),
    y_label: str | None = None,
) -> None:
    """Plot quality (one diversity metric) against measured wall-clock, per tool.

    Tools run over a budget series are drawn as a mean-over-seeds curve with a
    min/max band; single-shot tools as one marker at (mean time, mean quality).
    Reference lines are drawn dotted across the chart and reference markers as large
    crosses, each with a legend entry.

    Args:
        records: Run records for exactly one (problem, size) combination.
        metric_name: Which quality metric to plot (a key of ``RunRecord.quality``).
        path: Output webp file.
        title: Optional figure title.
        y_label: Axis label; defaults to `metric_name`.
    """
    import matplotlib.pyplot as plt  # deferred: keeps record-only workflows matplotlib-free

    use_docs_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    # Single-shot tools cycle through the marker shapes so tools whose markers land on the same
    # point stay distinguishable; budget-series tools cycle through the line styles.
    marker_index = 0
    series_index = 0
    for tool, tool_records in _legend_order(_group_by_tool(records)):
        color = tool_color(tool_key(tool))
        single_shot = all(r.budget == "single-shot" for r in tool_records)
        if single_shot:
            times = [r.measured_sec for r in tool_records]
            values = [r.quality[metric_name] for r in tool_records]
            marker = MARKER_SHAPES[marker_index % len(MARKER_SHAPES)]
            marker_index += 1
            ax.plot(np.mean(times), np.mean(values), marker, color=color, markersize=8, label=tool)
        else:
            t_mean, q_mean, q_min, q_max = _budget_series_stats(tool_records, metric_name)
            linestyle = _SERIES_LINESTYLES[series_index % len(_SERIES_LINESTYLES)]
            series_index += 1
            ax.plot(t_mean, q_mean, linestyle, marker=".", color=color, linewidth=_LINE_WIDTH, label=tool)
            ax.fill_between(t_mean, q_min, q_max, color=color, alpha=0.15, linewidth=0)
    for line in reference_lines:
        ax.axhline(line.value, color=line.color, linestyle=":", linewidth=1.2, label=line.label)
    for point in reference_markers:
        ax.plot(point.x_sec, point.value, point.marker, color=point.color, markersize=10, label=point.label)

    ax.set_xscale("log")
    ax.set_xlabel("measured wall-clock [s]")
    ax.set_ylabel(y_label or metric_name)
    _widen_narrow_y_axis(ax)
    # plain tick labels: matplotlib would otherwise factor a common prefix out of near-identical values
    ax.ticklabel_format(axis="y", useOffset=False)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, which="major")
    # Longer handles than the style sheet's: the per-point dot sits mid-handle and would hide the
    # one gap that tells the dashed multi-worker series from the solid single-worker one.
    ax.legend(handlelength=2.5)
    save_webp(fig, path)


def _widen_narrow_y_axis(ax) -> None:
    """Widen a y-axis narrower than `_MIN_Y_SPAN_FRACTION` of the median plotted value.

    Call it after every line is drawn: the median is taken over the lines on the axes, the
    reference lines included.
    """
    plotted = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()])
    y_low, y_high = ax.get_ylim()
    minimum_span = _MIN_Y_SPAN_FRACTION * abs(float(np.median(plotted)))
    if y_high - y_low < minimum_span:
        center = (y_low + y_high) / 2
        ax.set_ylim(center - minimum_span / 2, center + minimum_span / 2)


def _legend_order(by_tool: dict[str, list[RunRecord]]) -> list[tuple[str, list[RunRecord]]]:
    """Order max-div first (single worker before multi-worker), then the entrants alphabetically, then the random baseline."""
    return sorted(
        by_tool.items(),
        key=lambda item: (not item[0].startswith("max-div"), "workers" in item[0], item[0] == "random", item[0]),
    )


def _group_by_tool(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    """Group records by tool name."""
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.tool].append(rec)
    return dict(grouped)


def _budget_series_stats(
    records: list[RunRecord], metric_name: str
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Aggregate budget-series records per budget, in budget order: mean measured time, mean/min/max quality.

    The curve follows the budgets, not the measured times: a small budget whose set-up ran long
    would otherwise be drawn to the right of a larger one. `_charted_budgets` decides which
    budgets are drawn.
    """
    by_budget: dict[str, list[RunRecord]] = defaultdict(list)
    for rec in records:
        by_budget[rec.budget].append(rec)
    t_mean_by_budget = {budget: float(np.mean([r.measured_sec for r in recs])) for budget, recs in by_budget.items()}

    stats = []
    for budget in _charted_budgets(t_mean_by_budget):
        values = [r.quality[metric_name] for r in by_budget[budget]]
        stats.append((t_mean_by_budget[budget], float(np.mean(values)), min(values), max(values)))
    t_mean, q_mean, q_min, q_max = (list(component) for component in zip(*stats))
    return t_mean, q_mean, q_min, q_max


def _charted_budgets(t_mean_by_budget: dict[str, float]) -> list[str]:
    """Return the budgets to chart, in increasing order: each wall-clock budget must exceed the measured time of the previous charted one.

    A measured time is at least its budget, so the charted points have strictly increasing
    measured times, and a run of small budgets that all end at the set-up cost collapses to its
    first member. Iteration budgets are all charted.
    """
    charted: list[str] = []
    t_previous = -np.inf
    for budget in sorted(t_mean_by_budget, key=_budget_order):
        wall_clock = budget_sec(budget)
        if wall_clock is not None and wall_clock <= t_previous:
            continue
        charted.append(budget)
        t_previous = t_mean_by_budget[budget]
    return charted


def _budget_order(tag: str) -> float:
    """Return the sort key of a budget tag: the wall-clock budget, or the iteration count."""
    wall_clock = budget_sec(tag)
    return wall_clock if wall_clock is not None else float(iteration_count(tag))
