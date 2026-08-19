"""Anytime-curve figure: quality vs. measured wall-clock, budget series curves + single-shot dots."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.common.records import RunRecord


def plot_anytime_curve(records: list[RunRecord], metric_name: str, path: Path, title: str = "") -> None:
    """Plot quality (one diversity metric) against measured wall-clock, per tool.

    Budget-series tools (multiple budgets) are drawn as a mean-over-seeds curve with a
    min/max band; single-shot tools as one dot at (mean time, mean quality).

    Args:
        records: Run records for exactly one (problem, size) combination.
        metric_name: Which quality metric to plot (a key of ``RunRecord.quality``).
        path: Output file (suffix decides the format, e.g. ``.svg``/``.png``).
        title: Optional figure title.
    """
    import matplotlib.pyplot as plt  # deferred: keeps record-only workflows matplotlib-free

    fig, ax = plt.subplots(figsize=(8, 5))
    for tool, tool_records in sorted(_group_by_tool(records).items()):
        single_shot = all(r.budget == "single-shot" for r in tool_records)
        if single_shot:
            times = [r.measured_sec for r in tool_records]
            values = [r.quality[metric_name] for r in tool_records]
            ax.plot(np.mean(times), np.mean(values), "o", markersize=9, label=tool)
        else:
            t_mean, q_mean, q_min, q_max = _budget_series_stats(tool_records, metric_name)
            ax.plot(t_mean, q_mean, "-", marker=".", label=tool)
            ax.fill_between(t_mean, q_min, q_max, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("measured wall-clock [s]")
    ax.set_ylabel(metric_name)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
