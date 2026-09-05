"""Plot anytime curves: quality vs. measured wall-clock, budget-series curves + single-shot dots."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.common.records import RunRecord

from .style import MARKER_SHAPES, save_webp, tool_color, tool_key, use_docs_style

# Heavier than the scaling charts' lines: one curve per chart, at a smaller figure size.
_LINE_WIDTH = 1.5


def plot_anytime_curve(records: list[RunRecord], metric_name: str, path: Path, title: str = "") -> None:
    """Plot quality (one diversity metric) against measured wall-clock, per tool.

    Tools run over a budget series are drawn as a mean-over-seeds curve with a
    min/max band; single-shot tools as one marker at (mean time, mean quality).

    Args:
        records: Run records for exactly one (problem, size) combination.
        metric_name: Which quality metric to plot (a key of ``RunRecord.quality``).
        path: Output webp file.
        title: Optional figure title.
    """
    import matplotlib.pyplot as plt  # deferred: keeps record-only workflows matplotlib-free

    use_docs_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    # Single-shot tools cycle through the marker shapes so tools whose markers land on the same
    # point stay distinguishable.
    marker_index = 0
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
            ax.plot(t_mean, q_mean, "-", marker=".", color=color, linewidth=_LINE_WIDTH, label=tool)
            ax.fill_between(t_mean, q_min, q_max, color=color, alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("measured wall-clock [s]")
    ax.set_ylabel(metric_name)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, which="major")
    ax.legend()
    save_webp(fig, path)


def _legend_order(by_tool: dict[str, list[RunRecord]]) -> list[tuple[str, list[RunRecord]]]:
    """Order max-div first, then the competitors alphabetically, then the random baseline: the subject, the field, the floor."""
    return sorted(by_tool.items(), key=lambda item: (not item[0].startswith("max-div"), item[0] == "random", item[0]))


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
