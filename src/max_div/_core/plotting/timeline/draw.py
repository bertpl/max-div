"""Draw a `ParallelSolutionTimelinePlot` as a stacked-panel Matplotlib figure of the parallel solve.

The top panel shows the worker groups as horizontal bands over time: one block per group, stacked
with the shortest-lived groups on top and the surviving group at the bottom, each block as many bands
tall as the group's peak size. A band is colored:

- gray where no worker sits in it;
- cornflower where a worker does;
- light green while that worker's group holds the best score;
- a darker green on the one worker that held the best score at the time.

Below the top panel, the diversity trajectory gets its own panel, and the constraints trajectory a
third panel when it ever drops below one.

A saturating trajectory uses an upper-logarithmic y-axis that zooms in on where it flattens, fitted
to the trajectory sampled at uniform times, not to the checkpoints themselves: checkpoints crowd the
start of a solve, where improvements come fast, so fitting to them would fit the axis to the early
climb, not the long plateau that fills most of the picture.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from max_div._core.plotting.helpers import UpperLogTransform, figure_style

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

    from .model import GroupBlock, ParallelSolutionTimelinePlot, ScorePoint

# --- colors -------------------------------------------------
_EMPTY_BAND = "#ECEFF1"
_EMPTY_BAND_GONE = "#F6F7F8"  # ~50% from _EMPTY_BAND toward white, for a group after it has dissolved
_WORKER = "#4C72B0"
_BEST_GROUP = "#A9D5AC"
_BEST_WORKER = "#2E7D32"
_DIVERSITY = "#4C72B0"
_CONSTRAINTS = "#C44E52"
_DISSOLUTION_LINE = "#9E9E9E"  # faint vertical marker at each group dissolution, across every panel

# --- sizing (inches) ----------------------------------------
_FIG_WIDTH = 9.5
_BAND_INCH = 0.24
_BAND_SHRINK_THRESHOLD = 8  # bands shrink past this many, so the panel grows with sqrt of the band count
_SCORE_INCH = 2.28
_MARGIN_INCH = 0.9
_LABEL_SIZE = 6  # worker labels and score-panel tick labels share this size
_SCORE_TOP = 1.15  # top of the upper-log score axis, above the maximum at axis 1, so the headroom carries ticks
_GROUP_GAP = 0.5  # blank rows between one group's block and the next


def draw_timeline(plot: ParallelSolutionTimelinePlot) -> Figure:
    """Draw the timeline of a parallel solve and return the figure, styled like the docs figures."""
    with figure_style():
        return _draw_figure(plot)


def _draw_figure(plot: ParallelSolutionTimelinePlot) -> Figure:
    """Lay out the panels and draw each; kept separate so the whole figure is built under the style."""
    # shorter-lived groups on top, the surviving group at the bottom
    blocks = sorted(plot.group_blocks, key=lambda block: (block.t_end, block.group))
    base_y = _base_rows(blocks)
    total_extent = base_y[blocks[-1].group] + blocks[-1].height
    has_constraints_panel = any(point.constraints < 1.0 for point in plot.score_points)

    top_inch = _BAND_INCH * (
        total_extent if total_extent <= _BAND_SHRINK_THRESHOLD else (_BAND_SHRINK_THRESHOLD * total_extent) ** 0.5
    )
    heights = [top_inch, _SCORE_INCH] + ([_SCORE_INCH] if has_constraints_panel else [])
    fig = Figure(figsize=(_FIG_WIDTH, sum(heights) + _MARGIN_INCH))
    axes = fig.subplots(len(heights), 1, sharex=True, gridspec_kw={"height_ratios": heights, "hspace": 0.18})

    _draw_bands(axes[0], plot, blocks, base_y, total_extent)
    _draw_score(axes[1], plot, lambda point: point.diversity, _DIVERSITY, "diversity", is_log=True)
    if has_constraints_panel:
        # a constraints trajectory that reaches full satisfaction gets a plain linear axis; one that
        # never does gets the upper-log axis, which zooms in on its approach to the unreached limit
        has_full_constraints = max(point.constraints for point in plot.score_points) >= 1.0
        _draw_score(
            axes[2],
            plot,
            lambda point: point.constraints,
            _CONSTRAINTS,
            "constraints",
            is_log=not has_full_constraints,
        )

    dissolution_times = sorted({block.t_end for block in blocks if block.t_end < plot.t_end})
    for ax in axes:
        for t in dissolution_times:
            ax.axvline(t, color=_DISSOLUTION_LINE, linewidth=0.6, alpha=0.5, zorder=3)

    axes[-1].set_xlim(0.0, plot.t_end)
    axes[-1].set_xlabel("elapsed seconds")
    axes[0].set_title("Parallel solve timeline", fontsize=13, loc="left")
    fig.align_ylabels()
    return fig


# ==================================================================================================
#  Top panel: worker groups as bands
# ==================================================================================================
def _draw_bands(
    ax: Axes,
    plot: ParallelSolutionTimelinePlot,
    blocks: list[GroupBlock],
    base_y: dict[int, float],
    total_extent: float,
) -> None:
    """Draw the group blocks, the worker bands, the best-holder coloring and the preset labels."""
    for block in blocks:
        top = base_y[block.group]
        # the block spans the full width: normal gray while the group is alive, a much lighter gray
        # after it dissolves
        ax.add_patch(Rectangle((0.0, top), block.t_end, block.height, facecolor=_EMPTY_BAND, edgecolor="none"))
        if block.t_end < plot.t_end:
            ax.add_patch(
                Rectangle(
                    (block.t_end, top),
                    plot.t_end - block.t_end,
                    block.height,
                    facecolor=_EMPTY_BAND_GONE,
                    edgecolor="none",
                )
            )

    for segment in plot.segments:
        _draw_band(ax, segment.t_from, segment.t_to, base_y[segment.group] + segment.row, _WORKER)

    for interval in plot.best_intervals:
        for segment in plot.segments:
            if segment.group != interval.group:
                continue
            lo, hi = max(segment.t_from, interval.t_from), min(segment.t_to, interval.t_to)
            if lo < hi:
                _draw_band(
                    ax,
                    lo,
                    hi,
                    base_y[segment.group] + segment.row,
                    _BEST_WORKER if segment.worker == interval.worker else _BEST_GROUP,
                )

    _label_bands(ax, plot, base_y)
    ax.legend(
        handles=[
            Patch(facecolor=_WORKER, label="active worker"),
            Patch(facecolor=_BEST_GROUP, label="leading group"),
            Patch(facecolor=_BEST_WORKER, label="leading worker"),
        ],
        loc="upper right",
        fontsize=8,
    )
    ax.set_ylim(total_extent, 0)
    ax.set_yticks([])
    ax.set_ylabel("workers, grouped")


def _draw_band(ax: Axes, t_from: float, t_to: float, row: float, color: str) -> None:
    """Fill one band row from `t_from` to `t_to` with `color`."""
    ax.add_patch(Rectangle((t_from, row), t_to - t_from, 1.0, facecolor=color, edgecolor="none"))


def _label_bands(ax: Axes, plot: ParallelSolutionTimelinePlot, base_y: dict[int, float]) -> None:
    """Label each worker with its index and preset, just left of the band where it starts."""
    starts: dict[int, tuple[float, float]] = {}
    for segment in plot.segments:
        if (segment.worker not in starts) or (segment.t_from < starts[segment.worker][0]):
            starts[segment.worker] = (segment.t_from, base_y[segment.group] + segment.row)
    presets = plot.worker_presets
    for worker, (_t_from, row) in starts.items():
        # anchored at the left edge, not the worker's start, so the labels line up clear of the axis
        ax.text(0.0, row + 0.5, f"w{worker} {presets.get(worker, '')} ", ha="right", va="center", fontsize=_LABEL_SIZE)


def _base_rows(blocks: list[GroupBlock]) -> dict[int, float]:
    """Return the top row of each group's block, stacking them top-down with a blank gap between blocks."""
    base: dict[int, float] = {}
    offset = 0.0
    for block in blocks:
        base[block.group] = offset
        offset += block.height + _GROUP_GAP
    return base


# ==================================================================================================
#  Score panels
# ==================================================================================================
def _draw_score(
    ax: Axes,
    plot: ParallelSolutionTimelinePlot,
    value_of: Callable[[ScorePoint], float],
    color: str,
    label: str,
    is_log: bool,
) -> None:
    """Plot one score trajectory as a step curve, on an upper-log axis when `is_log`, else a linear one."""
    times = np.array([point.t for point in plot.score_points])
    values = np.array([value_of(point) for point in plot.score_points])

    if is_log:
        transform = UpperLogTransform.from_values(_values_at_uniform_times(times, values, plot.t_end))
        ax.plot(times, transform.to_axis(values), drawstyle="steps-post", color=color, linewidth=1.6)
        # the fit maps the maximum to axis 1; the ticks and gridlines run up to _SCORE_TOP above it,
        # so the plateau reads as reached, not pinned to the top edge
        positions, labels = _extended_log_ticks(transform, float(values.min()), 11)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.05, _SCORE_TOP)
    else:
        ax.plot(times, values, drawstyle="steps-post", color=color, linewidth=1.6)
        low = float(values.min())
        margin = max(0.01, 0.05 * (1.0 - low))
        ax.set_ylim(low - margin, 1.0 + margin)
    ax.grid(axis="y")
    ax.tick_params(axis="y", labelsize=_LABEL_SIZE)
    ax.set_ylabel(label)


def _values_at_uniform_times(times: np.ndarray, values: np.ndarray, t_end: float, n: int = 1000) -> np.ndarray:
    """Sample the step trajectory at `n` uniform times, so every stretch of the solve weighs the same in the fit."""
    grid = np.linspace(float(times[0]), t_end, n)
    index = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(values) - 1)
    return values[index]


def _extended_log_ticks(transform: UpperLogTransform, x_min: float, n: int) -> tuple[list[float], list[str]]:
    """Return `n` tick positions and labels spanning the minimum up to the value drawn at `_SCORE_TOP`.

    `transform.ticks_and_labels` only lays ticks out to the fitted maximum, at axis 1. This lays the
    same minimal-precision labels out over `[x_min, from_axis(_SCORE_TOP)]` instead — a second
    transform sharing `x_ref` puts that range on `[0, 1]` — then places each tick with `transform`, so
    the headroom above the maximum still carries ticks and gridlines.
    """
    x_top = float(transform.from_axis(_SCORE_TOP))
    c1 = 1.0 / (math.log(transform.x_ref - x_top) - math.log(transform.x_ref - x_min))
    c0 = -c1 * math.log(transform.x_ref - x_min)
    ticks, labels = UpperLogTransform(c0=c0, c1=c1, x_ref=transform.x_ref).ticks_and_labels(
        n, should_add_missing_ticks=True
    )
    return [float(transform.to_axis(float(tick))) for tick in ticks], labels
