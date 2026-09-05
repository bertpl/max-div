"""Plot anytime curves: quality vs. measured wall-clock, budget-series curves + single-shot dots + references."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.common.records import RunRecord

from .style import MARKER_SHAPES, save_webp, tool_color, tool_key, use_docs_style

# Heavier than the scaling charts' lines: one or two curves per chart, at a smaller figure size.
_LINE_WIDTH = 1.5
# Budget-series tools in legend order take these line styles in turn — the single-worker series
# solid, the multi-worker series dashed — so two max-div curves on one chart stay apart.
_SERIES_LINESTYLES = ("-", "--", "-.", ":")
_REFERENCE_COLOR = "#555555"


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
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
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
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, which="major")
    ax.legend()
    save_webp(fig, path)


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
    """Aggregate budget-series records per budget: mean measured time, mean/min/max quality."""
    by_budget: dict[str, list[RunRecord]] = defaultdict(list)
    for rec in records:
        by_budget[rec.budget].append(rec)

    stats = []
    for budget_records in by_budget.values():
        times = [r.measured_sec for r in budget_records]
        values = [r.quality[metric_name] for r in budget_records]
        stats.append((float(np.mean(times)), float(np.mean(values)), min(values), max(values)))
    stats.sort()
    t_mean, q_mean, q_min, q_max = (list(component) for component in zip(*stats))
    return t_mean, q_mean, q_min, q_max
