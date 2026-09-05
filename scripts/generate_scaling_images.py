"""Regenerate the solver-scaling result images and tables from the tracked measurement data.

Reads the stages' run records and the memory fits, and renders the combined time and memory
charts, one fit chart per fitted configuration, and the markdown tables and fragments the results
pages include — images under the scaling docs' `images/` folder (`IMAGES_DIR`), markdown under
`generated/`. Style sheet, tool colors, and the webp writer come from `benchmarks.figures.style`,
shared with the head-to-head anytime curves.

Run:  uv run --group benchmarks --python 3.14 python scripts/generate_scaling_images.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

# `benchmarks/` is repo-local rather than an installed package, so it is reached by putting the
# repo root on the path. Running the script from the repo root does not do that by itself.
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.figures.style import MARKER_SHAPES, save_webp, tool_color, use_docs_style  # noqa: E402
from benchmarks.solver_scaling.best_known_stage import DATA_PATH as BEST_KNOWN_DATA_PATH  # noqa: E402
from benchmarks.solver_scaling.best_known_stage import best_known_by_size  # noqa: E402
from benchmarks.solver_scaling.configs import CONFIGS  # noqa: E402
from benchmarks.solver_scaling.grid import (  # noqa: E402
    GRID_MIN,
    MEMORY_CAP_BYTES,
    REFERENCE_BUDGET_SEC,
    SETUP_GRACE_SEC,
    operational_bound,
    size_grid,
)
from benchmarks.solver_scaling.memory_fit import FIT_PATH  # noqa: E402
from benchmarks.solver_scaling.memory_stage import DATA_PATH as MEMORY_DATA_PATH  # noqa: E402
from benchmarks.solver_scaling.outcome import Outcome, classify  # noqa: E402
from benchmarks.solver_scaling.quality_stage import DATA_PATH as QUALITY_DATA_PATH  # noqa: E402
from benchmarks.solver_scaling.quality_stage import (  # noqa: E402
    NORMALIZED_QUALITY_THRESHOLDS,
    Q_RANDOM_PATH,
    best_known_pool,
    median_qualities,
    quality_limits,
)
from benchmarks.solver_scaling.records import ScalingRunRecord, load_scaling_records  # noqa: E402
from benchmarks.solver_scaling.time_stage import DATA_PATH as TIME_DATA_PATH  # noqa: E402
from benchmarks.solver_scaling.time_stage import passes_time  # noqa: E402

IMAGES_DIR = REPO_ROOT / "docs" / "benchmarks" / "third_party" / "scaling" / "images"
GENERATED_DIR = REPO_ROOT / "generated"
REGISTRY_FILE = REPO_ROOT / "data" / "solver_registry.yaml"

# One width for every plotted line, so the time chart's series and the memory chart's fit curves
# read the same weight.
_LINE_WIDTH = 1.0


# ==================================================================================================
#  Data access
# ==================================================================================================
def _solver_names() -> dict[str, str]:
    """Return the registry's tool key -> display name mapping."""
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {tool["key"]: tool["name"] for category in registry["categories"] for tool in category["tools"]}


def _records_by_config(records: list[ScalingRunRecord]) -> dict[tuple[str, str], list[ScalingRunRecord]]:
    """Group records per (tool, config), each group sorted by size, restricted to registry configurations."""
    grouped: dict[tuple[str, str], list[ScalingRunRecord]] = {(c.tool, c.name): [] for c in CONFIGS}
    for record in records:
        if (record.tool, record.config) in grouped:
            grouped[(record.tool, record.config)].append(record)
    return {key: sorted(rows, key=lambda r: r.n) for key, rows in grouped.items() if rows}


def _display_name(tool: str, names: dict[str, str]) -> str:
    """Return the tool's display name (its own column in the result tables)."""
    return names.get(tool, tool)


def _legend_label(tool: str, config: str, names: dict[str, str]) -> str:
    """Return a chart-legend label: the display name, with `[config]` appended when a tool has several.

    Brackets match the plain text a matplotlib legend renders (backticks would show literally);
    the result tables keep the configuration in its own column instead.
    """
    display = _display_name(tool, names)
    if len([c for c in CONFIGS if c.tool == tool]) == 1:
        return display
    return f"{display} [{config}]"


# ==================================================================================================
#  Charts
# ==================================================================================================
# The best-known and Q_random reference curves are not tools, so they carry their own colors.
_BEST_KNOWN_COLOR = "#4E8FD9"  # blue
_Q_RANDOM_COLOR = "#F29E4C"  # orange


def _series_style(tool: str, config: str) -> dict:
    """Return the plot kwargs (color, line style, marker) identifying one configuration on every chart.

    A tool's configurations share its color; marker shapes cycle over the configurations in
    `CONFIGS` order, every other one drawn open, so neighboring configurations differ in shape
    and fill both — the configurations that stop at their own budget all sit on the `T_max`
    line, where identical markers would hide one another.

    The style is keyed on the configuration itself, not on plot order: the renderers skip
    series with no plottable rows, so a position-based cycle would style the same configuration
    differently between the combined charts and the per-config fit charts.
    """
    tools = list(dict.fromkeys(c.tool for c in CONFIGS))
    tool_index = tools.index(tool)
    config_index = [(c.tool, c.name) for c in CONFIGS].index((tool, config))
    style = {
        "color": tool_color(tool),
        "linestyle": "-" if tool_index % 2 == 0 else "--",
        "marker": MARKER_SHAPES[config_index % len(MARKER_SHAPES)],
    }
    if config_index % 2:
        style.update(markerfacecolor="none", markersize=7)
    return style


def _grid_xticks(ax: plt.Axes, n_max: int, extra: tuple[int, ...] = ()) -> None:
    """Put ticks, labels and gridlines on every grid size up to `n_max`, plus any `extra` sizes."""
    ticks = sorted({*size_grid(n_max), *extra})
    ax.set_xticks(ticks)
    ax.set_xticklabels([_format_count(n) for n in ticks], rotation=45, ha="right", fontsize=8)
    ax.set_xticks([], minor=True)


def _format_count(n: int) -> str:
    """Format a grid size compactly (20, 50, ..., 1K, 2K, ..., 1M, ..., 1B)."""
    for suffix, factor in (("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if n >= factor:
            return f"{n // factor}{suffix}"
    return str(n)


def _format_bytes(value: float, _pos=None) -> str:
    """Format a byte count as a whole number of KB / MB / GB for axis ticks."""
    for unit, factor in (("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if value >= factor:
            return f"{value / factor:.0f} {unit}"
    return f"{value:.0f} B"


def render_time_chart(grouped: dict, names: dict[str, str]) -> None:
    """Render end-to-end solve time against problem size, one series per configuration."""
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    for (tool, config), rows in grouped.items():
        completed = [r for r in rows if r.completed and r.measured_sec is not None]
        if completed:
            ax.plot(
                [r.n for r in completed],
                [r.measured_sec for r in completed],
                label=_legend_label(tool, config, names),
                linewidth=_LINE_WIDTH,
                **_series_style(tool, config),
            )
    ax.axhline(REFERENCE_BUDGET_SEC, color="#888888", linestyle=":", linewidth=1.2)
    # x in axes fraction, y in data coordinates, so the label sits at the line's right end
    ax.text(
        0.99,
        REFERENCE_BUDGET_SEC * 1.3,
        "$\\mathrm{T_{max}}$ (1 min)",
        color="#555555",
        ha="right",
        va="bottom",
        fontsize=9,
        transform=ax.get_yaxis_transform(),
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    largest = max(r.n for rows in grouped.values() for r in rows if r.completed and r.measured_sec is not None)
    _grid_xticks(ax, largest)
    ax.set_xlabel("problem size n")
    ax.set_ylabel("end-to-end solve time [s]")
    ax.set_title("Solver Scaling — Time", fontweight="bold")
    ax.grid(True, which="major")
    ax.legend()
    save_webp(fig, IMAGES_DIR / "scaling_time.webp")


def render_memory_chart(grouped: dict, fits: dict, names: dict[str, str]) -> None:
    """Render recorded memory footprints against problem size, with each configuration's fitted curve overlaid."""
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    # Measurements are markers only and the fitted curve carries the line, so neither artist alone
    # shows a configuration's full style; the legend gets one proxy per configuration with both.
    legend_handles = []
    for (tool, config), rows in grouped.items():
        observed = _footprint_rows(rows)
        if not observed:
            continue
        style = _series_style(tool, config)
        ax.plot(
            [r.n for r in observed],
            [r.peak_memory_bytes for r in observed],
            **{**style, "linestyle": "none"},
        )
        _draw_fit_curve(ax, fits.get(f"{tool}/{config}", {}), observed, style)
        legend_handles.append(Line2D([], [], label=_legend_label(tool, config, names), linewidth=_LINE_WIDTH, **style))
    ax.axhline(MEMORY_CAP_BYTES, color="#888888", linestyle=":", linewidth=1.2)
    # x in axes fraction, y in data coordinates, so the label sits at the line's right end, just below it
    ax.text(
        0.99,
        MEMORY_CAP_BYTES / 1.15,
        "$\\mathrm{M_{max}}$",
        color="#555555",
        ha="right",
        va="top",
        fontsize=9,
        transform=ax.get_yaxis_transform(),
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _grid_xticks(ax, operational_bound(), extra=(GRID_MIN // 2, 5 * 10**9))  # one tick past each end of the grid
    ax.set_xlim(left=GRID_MIN / 5, right=6 * 10**9)  # room for the legend before the first grid size
    ax.set_xlabel("problem size n")
    ax.set_ylabel("peak memory use")
    # fixed factor-2 ticks in whole binary units; the default decade ticks label awkward values
    ax.set_yticks([2**e for e in range(27, 36)])
    ax.yaxis.set_major_formatter(FuncFormatter(_format_bytes))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_title("Solver Scaling — Memory", fontweight="bold")
    ax.grid(True, which="major")
    ax.legend(handles=legend_handles, loc="upper left")
    save_webp(fig, IMAGES_DIR / "scaling_memory.webp")


def _footprint_rows(rows: list[ScalingRunRecord]) -> list[ScalingRunRecord]:
    """Return the runs carrying a usable memory footprint: completed, or killed at the window's end.

    Worker-spawning configurations are excluded — they are not measured on the memory axis, so
    their runs must not appear as points or in the legend.
    """
    return [
        r
        for r in rows
        if r.peak_memory_bytes
        and not r.spawned_processes
        and classify(r.completed, r.reason) in (Outcome.SUCCESS, Outcome.TIMEOUT)
    ]


def _draw_fit_curve(ax: plt.Axes, fit: dict, observed: list[ScalingRunRecord], style: dict) -> None:
    """Draw a configuration's fitted footprint curve, from its measured sizes up to the memory cap.

    Args:
        style: The configuration's `_series_style`; the curve takes its color and line style, so
            the memory chart reads like the time chart.
    """
    coef = fit.get("coef")
    if not coef:
        return
    n_lo = observed[0].n
    n_hi = operational_bound()  # the largest grid size; extrapolating past it publishes nothing
    ns = np.geomspace(n_lo, n_hi, 200)
    predicted = sum(c * ns**p for p, c in enumerate(coef))
    visible = predicted <= 2 * MEMORY_CAP_BYTES  # stop shortly above the cap; further extrapolation says nothing
    ax.plot(
        ns[visible],
        predicted[visible],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=_LINE_WIDTH,
        alpha=0.8,
    )


def render_fit_charts(grouped: dict, fits: dict, names: dict[str, str]) -> None:
    """Render one chart per fitted configuration and the markdown fragment embedding them.

    The combined chart's log y-scale flattens most series, so each configuration also gets its
    own chart on an adaptive linear scale — just its footprints and fitted curve, annotated with
    the fitted coefficients and R^2 — making visible that the fit follows a trend in the data.
    """
    thumbnails: list[str] = []
    written: set[Path] = set()
    for (tool, config), rows in grouped.items():
        fit = fits.get(f"{tool}/{config}", {})
        observed = _footprint_rows(rows)
        if not observed or not fit.get("coef"):
            continue
        style = _series_style(tool, config)  # the combined chart's color and marker for this config
        color = style["color"]
        label = _legend_label(tool, config, names)
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(
            [r.n for r in observed],
            [r.peak_memory_bytes / 2**20 for r in observed],
            **{**style, "linestyle": "none"},
        )
        coef = fit["coef"]
        fit_ns = np.geomspace(GRID_MIN, observed[-1].n, 200)
        predicted = sum(c * fit_ns**p for p, c in enumerate(coef)) / 2**20
        ax.plot(fit_ns, predicted, linestyle=style["linestyle"], linewidth=1.2, color=color)
        ax.text(
            0.03,
            0.94,
            f"{_format_fit(coef)}\n$R^2$ = {fit['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.set_xscale("log")
        _grid_xticks(ax, observed[-1].n)
        ax.set_xlabel("problem size n")
        ax.set_ylabel("peak memory use [MB]")
        ax.set_title(f"Memory fit — {label}", fontweight="bold")
        ax.grid(True, which="major")
        name = f"scaling_memory_fit_{tool}_{config}.webp"
        save_webp(fig, IMAGES_DIR / name)
        written.add(IMAGES_DIR / name)
        # raw HTML paths are not rewritten by mkdocs (unlike markdown image syntax), so they must
        # be relative to the page's built directory URL, one level below the section
        thumbnails.append(
            f'<a href="../images/{name}"><img src="../images/{name}" '
            f'alt="Memory footprints and fitted curve for {label}" width="32%"></a>'
        )
    # A configuration that stops getting a fit (e.g. it now hits the cap or fails instead) leaves a
    # stale chart behind, since a run only ever writes files; drop any fit chart not written this run.
    for stale in set(IMAGES_DIR.glob("scaling_memory_fit_*.webp")) - written:
        stale.unlink()
        print(f"removed {stale.relative_to(REPO_ROOT)}")
    # The fragment lays three thumbnails per row, each linking to the full-size chart.
    _write_generated("scaling_memory_fits.md", [" ".join(thumbnails)] if thumbnails else [])


def _format_fit(coef: tuple) -> str:
    """Format a fitted model as `f(n) = <c0> + <c1> B·n [+ <c2> B·n²]`."""
    terms = [_format_bytes(coef[0]), f"{coef[1]:.1f} B·n"]
    if len(coef) > 2:
        terms.append(f"{coef[2]:.1f} B·n²")
    return "f(n) = " + " + ".join(terms)


# Per-point label geometry for the best-known chart. The solver name (bold, dark) stacks one text
# line above its config (lighter gray), both rotated together so the two-line block runs up and to
# the right of its marker.
_BEST_KNOWN_LABEL_ROTATION = 30
_BEST_KNOWN_LABEL_FONTSIZE = 7


def _annotate_best_known_point(ax: plt.Axes, x: float, y: float, solver: str, config: str) -> None:
    """Annotate one best-known point with the bold solver name over its config name.

    Two short lines with contrasting weight and color tell the solver and config apart, which a
    single run-on `solver [config]` label in a single weight and color would not — and each line is far shorter
    than the joined one, so the rotated labels crowd the curve less.
    """
    theta = math.radians(_BEST_KNOWN_LABEL_ROTATION)
    # step one text line up along the rotated text's normal, so the solver name sits above the config
    line_gap = _BEST_KNOWN_LABEL_FONTSIZE + 3
    up = (-math.sin(theta) * line_gap, math.cos(theta) * line_gap)
    shared = {
        "textcoords": "offset points",
        "ha": "left",
        "va": "bottom",
        "fontsize": _BEST_KNOWN_LABEL_FONTSIZE,
        "rotation": _BEST_KNOWN_LABEL_ROTATION,
        "rotation_mode": "anchor",
    }
    ax.annotate(f"[{config}]", (x, y), xytext=(2, 4), color="#666666", **shared)
    ax.annotate(solver, (x, y), xytext=(2 + up[0], 4 + up[1]), color="#333333", fontweight="bold", **shared)


def _label_curve_end(ax: plt.Axes, x: float, y: float, text: str, offset: tuple[float, float] = (6, 0)) -> None:
    """Write `text` right of a curve's last point, `offset` points away from it, vertically centered."""
    ax.annotate(
        text, (x, y), xytext=offset, textcoords="offset points", ha="left", va="center", fontsize=9, color="#555555"
    )


def render_best_known_chart(
    records: list[ScalingRunRecord],
    quality_records: list[ScalingRunRecord],
    q_random: dict[int, float],
    names: dict[str, str],
) -> None:
    """Render the best-known diversity and the random reference against problem size (log--log).

    The two curves are the band the normalized quality is measured against: the best-known
    curve (the highest quality any run reached at each size) on top, the random reference below.
    Each best-known point is labeled with the tool that produced it and its winning configuration.
    Every configuration's median quality (`quality_records`) is drawn as faint context, and the
    verdict thresholds as dotted lines between the two references.
    """
    by_size = best_known_by_size(records)
    sizes = sorted(by_size)
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    # Drawn first, so the references and their labels stay on top.
    for index, medians in enumerate(median_qualities(quality_records).values()):
        median_sizes = sorted(medians)
        ax.plot(
            median_sizes,
            [medians[n] for n in median_sizes],
            color="#D8D8D8",
            linewidth=0.5,
            zorder=1,
            label="results per solver config (median)" if index == 0 else None,  # one legend entry for all
        )
    ax.plot(
        sizes,
        [by_size[n].min_separation for n in sizes],
        label="best-known",
        linewidth=_LINE_WIDTH,
        color=_BEST_KNOWN_COLOR,
        marker="o",
    )
    for n in sizes:
        record = by_size[n]
        _annotate_best_known_point(ax, n, record.min_separation, _display_name(record.tool, names), record.config)
    random_sizes = [n for n in sizes if n in q_random]
    ax.plot(
        random_sizes,
        [q_random[n] for n in random_sizes],
        label="$Q_{\\mathrm{random}}$",
        linewidth=_LINE_WIDTH,
        color=_Q_RANDOM_COLOR,
        marker="s",
        markerfacecolor="none",
        markersize=7,
    )
    # Each curve and threshold line ends in its normalized-quality value, the scale the quality page judges on.
    for index, threshold in enumerate(NORMALIZED_QUALITY_THRESHOLDS):
        threshold_values = [(1 - threshold) * q_random[n] + threshold * by_size[n].min_separation for n in random_sizes]
        ax.plot(
            random_sizes,
            threshold_values,
            color="#888888",
            linestyle=":",
            linewidth=1.0,
            label="scaling thresholds" if index == 0 else None,  # one legend entry for all
        )
        _label_curve_end(ax, random_sizes[-1], threshold_values[-1], f"{threshold:.0%}", offset=(2, -5.5))
    _label_curve_end(ax, random_sizes[-1], q_random[random_sizes[-1]], "0%")
    _label_curve_end(ax, sizes[-1], by_size[sizes[-1]].min_separation, "100%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)  # headroom above the top curve for its rotated labels
    ax.set_xlim(right=ax.get_xlim()[1] * 1.5)  # and to the right, so the last size's label fits
    _grid_xticks(ax, max(sizes))
    ax.set_xlabel("problem size n")
    ax.set_ylabel("diversity (minimum separation)")
    ax.set_title("Solver Scaling — Best-Known Quality", fontweight="bold")
    ax.grid(True, which="major")
    # Order the legend entries top to bottom as the chart stacks them.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    order = ["best-known", "scaling thresholds", "results per solver config (median)", "$Q_{\\mathrm{random}}$"]
    ax.legend([by_label[label] for label in order], order)
    save_webp(fig, IMAGES_DIR / "scaling_best_known.webp")


def render_normalized_quality_chart(
    quality_records: list[ScalingRunRecord],
    best_known_records: list[ScalingRunRecord],
    q_random: dict[int, float],
    names: dict[str, str],
) -> None:
    """Render each configuration's normalized solution quality against problem size (log x, linear y).

    Gray dotted lines mark the scale ends and the verdict thresholds; a configuration below the random
    reference plots a negative percentage, so the y-axis is left free to descend below zero.
    """
    pool = best_known_pool(quality_records, best_known_records)
    medians = median_qualities(quality_records)
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    for (tool, config), by_size in medians.items():
        points = [
            (n, _normalized_quality(median, q_random[n], pool[n]) * 100)
            for n, median in sorted(by_size.items())
            if n in pool and n in q_random
        ]
        if points:
            ax.plot(
                [n for n, _ in points],
                [percent for _, percent in points],
                label=_legend_label(tool, config, names),
                linewidth=_LINE_WIDTH,
                **_series_style(tool, config),
            )
    for level in (0, 50, 90, 100):
        ax.axhline(level, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    _grid_xticks(ax, max(pool))
    ax.set_xlabel("problem size n")
    ax.set_ylabel("normalized solution quality")
    ax.set_yticks(range(-10, 101, 10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    ax.set_title("Solver Scaling — Normalized Quality", fontweight="bold")
    ax.grid(True, which="major", axis="x")
    ax.legend(loc="center right")
    save_webp(fig, IMAGES_DIR / "scaling_normalized_quality.webp")


# ==================================================================================================
#  Tables
# ==================================================================================================
def _sweep_end(rows: list[ScalingRunRecord]) -> str:
    """Describe what ended a configuration's size sweep, from its largest attempted size's record."""
    last = rows[-1]
    if passes_time(last):
        return "grid exhausted"
    outcome = classify(last.completed, last.reason)
    if outcome is Outcome.SCALING_FAILURE:
        return f"failure at n={last.n:,}: `{last.reason}`"
    if outcome is Outcome.MEMORY:
        return f"memory cap exceeded at n={last.n:,}"
    # A run over the time budget gets one message with the time shown, whether it finished within
    # the kill grace (measured time known) or was killed at the deadline (time only bounded below).
    if last.completed and last.measured_sec is not None:
        elapsed = f"{last.measured_sec:.0f} s"
    else:
        elapsed = f"≥{int(REFERENCE_BUDGET_SEC + SETUP_GRACE_SEC)} s"
    return f"time budget exceeded at n={last.n:,} ({elapsed})"


def write_time_table(grouped: dict, names: dict[str, str]) -> None:
    """Write the per-configuration largest-n-within-time table."""
    lines = [
        "| Solver | Config | Largest n within the time budget | Sweep ended by |",
        "|---|---|---|---|",
    ]
    for (tool, config), rows in grouped.items():
        passing = [r.n for r in rows if passes_time(r)]
        largest = f"**{max(passing):,}**" if passing else "—"
        lines.append(f"| {_display_name(tool, names)} | `{config}` | {largest} | {_sweep_end(rows)} |")
    _write_generated("scaling_time.md", lines)


def write_memory_table(fits: dict, names: dict[str, str]) -> None:
    """Write the per-configuration largest-n-within-memory table, one row per fit entry."""
    lines = [
        "| Solver | Config | Largest n within memory | Determination |",
        "|---|---|---|---|",
    ]
    for key, fit in fits.items():
        tool, config = key.split("/", 1)
        largest = f"**{fit['max_n']:,}**" if fit.get("max_n") else "—"
        lines.append(f"| {_display_name(tool, names)} | `{config}` | {largest} | {fit.get('reason', 'no fit')} |")
    _write_generated("scaling_memory.md", lines)


def write_best_known_table(records: list[ScalingRunRecord], q_random: dict[int, float], names: dict[str, str]) -> None:
    """Write the best-known-solution provenance table: per size, the best result and the bounds.

    Beside each size's best-known value sit `Q_random` and the two verdict thresholds derived
    from the pair, so the table shows the whole scale a size's verdicts are judged on. The
    measured-time column keeps a late completion visible — the best-known stage keeps a run
    that finished past its budget (see `best_known_stage`).
    """
    lines = [
        "| Problem size n | Q_random | 50% threshold | 90% threshold"
        " | Best-known quality | Solver | Config | Measured time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, record in best_known_by_size(records).items():
        random_quality = q_random.get(n)
        thresholds = [
            f"{(1 - b) * random_quality + b * record.min_separation:.4f}" if random_quality is not None else ""
            for b in sorted(NORMALIZED_QUALITY_THRESHOLDS)
        ]
        random_cell = f"{random_quality:.4f}" if random_quality is not None else ""
        lines.append(
            f"| {n:,} | {random_cell} | {thresholds[0]} | {thresholds[1]}"
            f" | {record.min_separation:.4f} | {_display_name(record.tool, names)}"
            f" | `{record.config}` | {record.measured_sec:.0f} s |"
        )
    _write_generated("scaling_best_known.md", lines)


def write_quality_table(
    quality_records: list[ScalingRunRecord],
    best_known_records: list[ScalingRunRecord],
    q_random: dict[int, float],
    names: dict[str, str],
) -> None:
    """Write the per-configuration quality-limit table, one column per normalized-quality threshold."""
    per_threshold = [
        quality_limits(quality_records, best_known_records, q_random, threshold)
        for threshold in NORMALIZED_QUALITY_THRESHOLDS
    ]
    lines = [
        "| Solver | Config | "
        + " | ".join(f"Largest n with normalized quality ≥ {t:.0%}" for t in NORMALIZED_QUALITY_THRESHOLDS)
        + " |",
        "|---|---|" + "---|" * len(NORMALIZED_QUALITY_THRESHOLDS),
    ]
    for key in per_threshold[0]:
        tool, config = key.split("/", 1)
        cells = " | ".join(f"**{limits[key]:,}**" if limits[key] else "—" for limits in per_threshold)
        lines.append(f"| {_display_name(tool, names)} | `{config}` | {cells} |")
    _write_generated("scaling_quality.md", lines)


def write_normalized_quality_table(
    quality_records: list[ScalingRunRecord],
    best_known_records: list[ScalingRunRecord],
    q_random: dict[int, float],
    names: dict[str, str],
) -> None:
    """Write the normalized-quality table: per configuration and size, the median's normalized solution quality.

    A cell holds `(Q_median - Q_random) / (Q_best_known - Q_random)`; the verdict criterion is the
    same value reaching a `NORMALIZED_QUALITY_THRESHOLDS` entry: a cell reaching the strictest one
    is bold, one reaching only the lowest is italic. An empty cell is a size the configuration was
    not judged at (beyond its time limit, or no completed run).
    """
    pool = best_known_pool(quality_records, best_known_records)
    medians = median_qualities(quality_records)
    sizes = size_grid(max(pool)) if pool else []
    lines = [
        "| Solver | Config | " + " | ".join(_format_count(n) for n in sizes) + " |",
        "|---|---|" + "---|" * len(sizes),
    ]
    for (tool, config), by_size in medians.items():
        cells = []
        for n in sizes:
            median = by_size.get(n)
            if median is None:
                cells.append("")
                continue
            cells.append(_format_normalized_quality(median, q_random[n], pool[n]))
        lines.append(f"| {_display_name(tool, names)} | `{config}` | " + " | ".join(cells) + " |")
    _write_generated("scaling_normalized_quality.md", lines)


def _normalized_quality(median: float, random_quality: float, best_known: float) -> float:
    """Return a median quality on the normalized scale: 0 at the random reference, 1 at the best-known.

    A degenerate size where the best-known equals the random reference has no scale to normalize
    on; matching the best-known is then the only way to reach 1.
    """
    span = best_known - random_quality
    return (median - random_quality) / span if span > 0 else (1.0 if median >= best_known else 0.0)


def _format_normalized_quality(median: float, random_quality: float, best_known: float) -> str:
    """Format one normalized-quality cell as a percentage: bold when it reaches the strictest
    `NORMALIZED_QUALITY_THRESHOLDS` entry, italic when it reaches only the lowest.

    The display rounds down to 0.1% while the marking judges the exact value, so a printed
    50.0% is genuinely at or above the threshold and the marking never contradicts the number.
    """
    quality = _normalized_quality(median, random_quality, best_known)
    percent = math.floor(quality * 1000) / 10
    if quality >= max(NORMALIZED_QUALITY_THRESHOLDS):
        return f"**{percent:.1f}%**"
    if quality >= min(NORMALIZED_QUALITY_THRESHOLDS):
        return f"*{percent:.1f}%*"
    return f"{percent:.1f}%"


def _write_generated(name: str, lines: list[str]) -> None:
    """Write one generated markdown fragment."""
    path = GENERATED_DIR / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


# ==================================================================================================
#  Main entrypoint
# ==================================================================================================
def main() -> None:
    """Regenerate the charts and tables from the stages' tracked data files."""
    use_docs_style()
    names = _solver_names()
    time_grouped = _records_by_config(load_scaling_records(TIME_DATA_PATH) if TIME_DATA_PATH.exists() else [])
    memory_grouped = _records_by_config(load_scaling_records(MEMORY_DATA_PATH) if MEMORY_DATA_PATH.exists() else [])
    fits = json.loads(FIT_PATH.read_text(encoding="utf-8")) if FIT_PATH.exists() else {}
    best_known_records = load_scaling_records(BEST_KNOWN_DATA_PATH) if BEST_KNOWN_DATA_PATH.exists() else []
    quality_records = load_scaling_records(QUALITY_DATA_PATH) if QUALITY_DATA_PATH.exists() else []
    q_random_values = (
        {int(n): v for n, v in json.loads(Q_RANDOM_PATH.read_text(encoding="utf-8")).items()}
        if Q_RANDOM_PATH.exists()
        else {}
    )
    if best_known_records:
        # The quality runs join the pool: a reference-budget run can hold a size's best solution.
        pool_records = best_known_records + quality_records
        write_best_known_table(pool_records, q_random_values, names)
        render_best_known_chart(pool_records, quality_records, q_random_values, names)
    if quality_records and q_random_values:
        write_quality_table(quality_records, best_known_records, q_random_values, names)
        write_normalized_quality_table(quality_records, best_known_records, q_random_values, names)
        render_normalized_quality_chart(quality_records, best_known_records, q_random_values, names)
    render_time_chart(time_grouped, names)
    render_memory_chart(memory_grouped, fits, names)
    render_fit_charts(memory_grouped, fits, names)
    write_time_table(time_grouped, names)
    write_memory_table(fits, names)


if __name__ == "__main__":
    main()
