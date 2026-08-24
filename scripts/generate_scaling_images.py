"""Regenerate the solver-scaling result images and tables from the tracked measurement data.

Reads the two sweeps' run records and the memory fits, and renders the combined time and memory
charts, one fit chart per fitted configuration, and the markdown tables and fragments the results
pages include — images under the scaling docs' `images/` folder (`IMAGES_DIR`), markdown under
`generated/`. Charts use the docs Matplotlib style sheet (`STYLE_SHEET`), shared with the
benchmark-problem and preset-results figures.

Run:  uv run --group benchmarks --python 3.13 python scripts/generate_scaling_images.py
"""

import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FuncFormatter
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

# `benchmarks/` is repo-local rather than an installed package, so it is reached by putting the
# repo root on the path. Running the script from the repo root does not do that by itself.
sys.path.insert(0, str(REPO_ROOT))

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
from benchmarks.solver_scaling.records import ScalingRunRecord, load_scaling_records  # noqa: E402
from benchmarks.solver_scaling.time_stage import DATA_PATH as TIME_DATA_PATH  # noqa: E402
from benchmarks.solver_scaling.time_stage import passes_time  # noqa: E402

IMAGES_DIR = REPO_ROOT / "docs" / "benchmarks" / "third_party" / "scaling" / "images"
GENERATED_DIR = REPO_ROOT / "generated"
STYLE_SHEET = REPO_ROOT / "local" / "docs" / "figures" / "docs.mplstyle"
REGISTRY_FILE = REPO_ROOT / "data" / "solver_registry.yaml"


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
def _save_webp(fig: plt.Figure, path: Path) -> None:
    """Render the figure to quality-92 webp via PIL (matplotlib has no native webp writer)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(buffer).convert("RGB").save(path, format="WEBP", lossless=False, quality=92)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def _series_marker(index: int) -> dict:
    """Return per-series marker kwargs: shapes alternate and every other series is drawn open.

    Distinct shapes and open-vs-filled markers keep coinciding series tellable apart — the
    self-limiting configurations all sit on the `T_max` line, where identical dots would
    hide one another completely.
    """
    shapes = ("o", "s", "D", "^", "v", "P", "X")
    marker = shapes[index % len(shapes)]
    if index % 2:
        return {"marker": marker, "markerfacecolor": "none", "markersize": 7}
    return {"marker": marker}


def _grid_xticks(ax: plt.Axes, n_max: int) -> None:
    """Put ticks, labels and gridlines on every grid size up to `n_max`."""
    ticks = size_grid(n_max)
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
    for index, ((tool, config), rows) in enumerate(grouped.items()):
        completed = [r for r in rows if r.completed and r.measured_sec is not None]
        if completed:
            ax.plot(
                [r.n for r in completed],
                [r.measured_sec for r in completed],
                label=_legend_label(tool, config, names),
                linewidth=1.4,
                **_series_marker(index),
            )
    ax.axhline(REFERENCE_BUDGET_SEC, color="#888888", linestyle="--", linewidth=1.2)
    # x in axes fraction, y in data coordinates, so the label hugs the line's right end
    ax.text(
        0.99,
        REFERENCE_BUDGET_SEC * 1.2,
        "T_max (1 min)",
        color="#555555",
        ha="right",
        fontsize=9,
        transform=ax.get_yaxis_transform(),
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
    _save_webp(fig, IMAGES_DIR / "scaling_time.webp")


def render_memory_chart(grouped: dict, fits: dict, names: dict[str, str]) -> None:
    """Render recorded memory footprints against problem size, with each configuration's fitted curve overlaid."""
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    for index, ((tool, config), rows) in enumerate(grouped.items()):
        observed = _footprint_rows(rows)
        if not observed:
            continue
        handles = ax.plot(
            [r.n for r in observed],
            [r.peak_memory_bytes for r in observed],
            linestyle="none",
            label=_legend_label(tool, config, names),
            **_series_marker(index),
        )
        _draw_fit_curve(ax, fits.get(f"{tool}/{config}", {}), observed, handles[0].get_color())
    ax.axhline(MEMORY_CAP_BYTES, color="#888888", linestyle="--", linewidth=1.2)
    # x in axes fraction, y in data coordinates, so the label hugs the line's right end
    ax.text(
        0.99,
        MEMORY_CAP_BYTES * 1.3,
        "M_max (32 GB)",
        color="#555555",
        ha="right",
        fontsize=9,
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _grid_xticks(ax, operational_bound())
    ax.set_xlabel("problem size n")
    ax.set_ylabel("peak memory use")
    # fixed factor-4 ticks in whole binary units; the default decade ticks label awkward values
    ax.set_yticks([2**27, 2**29, 2**31, 2**33, 2**35])
    ax.yaxis.set_major_formatter(FuncFormatter(_format_bytes))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_title("Solver Scaling — Memory", fontweight="bold")
    ax.grid(True, which="major")
    ax.legend(loc="upper left")
    _save_webp(fig, IMAGES_DIR / "scaling_memory.webp")


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


def _draw_fit_curve(ax: plt.Axes, fit: dict, observed: list[ScalingRunRecord], color: str) -> None:
    """Draw a configuration's fitted footprint curve, from its measured sizes up to the memory cap."""
    coef = fit.get("coef")
    if not coef:
        return
    n_lo = observed[0].n
    n_hi = operational_bound()  # the largest grid size; extrapolating past it publishes nothing
    ns = np.geomspace(n_lo, n_hi, 200)
    predicted = sum(c * ns**p for p, c in enumerate(coef))
    visible = predicted <= 2 * MEMORY_CAP_BYTES  # stop shortly above the cap; further extrapolation says nothing
    ax.plot(ns[visible], predicted[visible], color=color, linestyle="--", linewidth=1.0, alpha=0.8)


def render_fit_charts(grouped: dict, fits: dict, names: dict[str, str]) -> None:
    """Render one chart per fitted configuration and the markdown fragment embedding them.

    The combined chart's log y-scale flattens most series, so each configuration also gets its
    own chart on an adaptive linear scale — just its footprints and fitted curve, annotated with
    the fitted coefficients and R^2 — making visible that the fit follows a trend in the data.
    """
    thumbnails: list[str] = []
    written: set[Path] = set()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, ((tool, config), rows) in enumerate(grouped.items()):
        fit = fits.get(f"{tool}/{config}", {})
        observed = _footprint_rows(rows)
        if not observed or not fit.get("coef"):
            continue
        # index into the same cycle as the combined chart's series order, so each configuration
        # keeps one color across all memory charts
        color = colors[index % len(colors)]
        label = _legend_label(tool, config, names)
        # the combined chart's marker for this config, so shape and fill match its color there
        marker_style = _series_marker(index)
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(
            [r.n for r in observed],
            [r.peak_memory_bytes / 2**20 for r in observed],
            linestyle="none",
            color=color,
            **marker_style,
        )
        coef = fit["coef"]
        fit_ns = np.geomspace(GRID_MIN, observed[-1].n, 200)
        predicted = sum(c * fit_ns**p for p, c in enumerate(coef)) / 2**20
        ax.plot(fit_ns, predicted, linestyle="--", linewidth=1.2, color=color)
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
        _save_webp(fig, IMAGES_DIR / name)
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
    # A run over the time budget, whether it finished within the kill grace (measured time known)
    # or was killed at the deadline (time only bounded below) — one message, with the time shown.
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


def _write_generated(name: str, lines: list[str]) -> None:
    """Write one generated markdown fragment."""
    path = GENERATED_DIR / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


# ==================================================================================================
#  Main entrypoint
# ==================================================================================================
def main() -> None:
    """Regenerate the charts and tables from the two sweeps' tracked data files."""
    plt.style.use(STYLE_SHEET)
    names = _solver_names()
    time_grouped = _records_by_config(load_scaling_records(TIME_DATA_PATH) if TIME_DATA_PATH.exists() else [])
    memory_grouped = _records_by_config(load_scaling_records(MEMORY_DATA_PATH) if MEMORY_DATA_PATH.exists() else [])
    fits = json.loads(FIT_PATH.read_text(encoding="utf-8")) if FIT_PATH.exists() else {}
    render_time_chart(time_grouped, names)
    render_memory_chart(memory_grouped, fits, names)
    render_fit_charts(memory_grouped, fits, names)
    write_time_table(time_grouped, names)
    write_memory_table(fits, names)


if __name__ == "__main__":
    main()
