"""Regenerate the solver-scaling result images and tables from the tracked measurement data.

Reads the run records and memory fits under ``benchmarks/solver_scaling/data/`` and renders:

* ``docs/benchmarks/third_party/scaling/images/scaling_time.webp`` — end-to-end solve time
  against problem size, with the time budget drawn in;
* ``docs/benchmarks/third_party/scaling/images/scaling_memory.webp`` — recorded peak RSS against
  problem size, each configuration's fitted growth curve overlaid, with the memory cap drawn in;
* ``generated/scaling_time.md`` / ``generated/scaling_memory.md`` — the per-configuration result
  tables the results pages include.

Charts use the docs Matplotlib style sheet (``local/docs/figures/docs.mplstyle``), shared with
the benchmark-problem and preset-results figures.

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
    MEMORY_CAP_BYTES,
    REFERENCE_BUDGET_SEC,
    operational_bound,
    size_grid,
)
from benchmarks.solver_scaling.memory_fit import FIT_PATH  # noqa: E402
from benchmarks.solver_scaling.outcome import Outcome, classify  # noqa: E402
from benchmarks.solver_scaling.records import ScalingRunRecord, load_scaling_records  # noqa: E402
from benchmarks.solver_scaling.time_stage import DATA_PATH, passes_time  # noqa: E402

IMAGES_DIR = REPO_ROOT / "docs" / "benchmarks" / "third_party" / "scaling" / "images"
GENERATED_DIR = REPO_ROOT / "generated"
STYLE_SHEET = REPO_ROOT / "local" / "docs" / "figures" / "docs.mplstyle"
REGISTRY_FILE = REPO_ROOT / "data" / "solver_registry.yaml"

_SWEEP_END_LABELS = {
    Outcome.TIMEOUT: "time budget exceeded",
    Outcome.MEMORY: "memory cap exceeded",
}


# =================================================================================================
#  Data access
# =================================================================================================
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


def _series_label(tool: str, config: str, names: dict[str, str], markdown: bool = False) -> str:
    """Return a series label: the display name, with the configuration appended when a tool has several.

    With `markdown` the configuration renders as inline code, matching the solver-configurations
    page; chart legends get the plain form, since matplotlib renders backticks literally.
    """
    display = names.get(tool, tool)
    if len([c for c in CONFIGS if c.tool == tool]) == 1:
        return display
    return f"{display} `{config}`" if markdown else f"{display} {config}"


# =================================================================================================
#  Charts
# =================================================================================================
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
    budget-honoring configurations all sit on the `T_max` line, where identical dots would
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
                label=_series_label(tool, config, names),
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
    """Render recorded peak RSS against problem size, with each configuration's fitted curve overlaid."""
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    for index, ((tool, config), rows) in enumerate(grouped.items()):
        completed = [r for r in rows if r.completed and r.peak_rss_bytes]
        if not completed:
            continue
        handles = ax.plot(
            [r.n for r in completed],
            [r.peak_rss_bytes for r in completed],
            linestyle="none",
            label=_series_label(tool, config, names),
            **_series_marker(index),
        )
        _draw_fit_curve(ax, fits.get(f"{tool}/{config}", {}), completed, handles[0].get_color())
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
    ax.set_ylabel("peak RSS")
    # fixed factor-4 ticks in whole binary units; the default decade ticks label awkward values
    ax.set_yticks([2**27, 2**29, 2**31, 2**33, 2**35])
    ax.yaxis.set_major_formatter(FuncFormatter(_format_bytes))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_title("Solver Scaling — Memory", fontweight="bold")
    ax.grid(True, which="major")
    ax.legend(loc="upper left")
    _save_webp(fig, IMAGES_DIR / "scaling_memory.webp")


def _draw_fit_curve(ax: plt.Axes, fit: dict, completed: list[ScalingRunRecord], color: str) -> None:
    """Draw a configuration's fitted peak-RSS curve, from its measured sizes up to the memory cap."""
    coef = fit.get("coef")
    if not coef:
        return
    n_lo = completed[0].n
    n_hi = operational_bound()  # the largest grid size; extrapolating past it publishes nothing
    ns = np.geomspace(n_lo, n_hi, 200)
    predicted = sum(c * ns**p for p, c in enumerate(coef))
    visible = predicted <= 2 * MEMORY_CAP_BYTES  # stop shortly above the cap; further extrapolation says nothing
    ax.plot(ns[visible], predicted[visible], color=color, linestyle="--", linewidth=1.0, alpha=0.8)


# =================================================================================================
#  Tables
# =================================================================================================
def _sweep_end(rows: list[ScalingRunRecord]) -> str:
    """Describe what ended a configuration's size sweep, from its largest attempted size's record."""
    last = rows[-1]
    if passes_time(last):
        return "grid exhausted"
    outcome = classify(last.completed, last.reason)
    if outcome is Outcome.SUCCESS:  # completed, but past the time budget
        return f"completed past the time budget at n={last.n:,} ({last.measured_sec:.0f} s)"
    if outcome is Outcome.SCALING_FAILURE:
        return f"failure at n={last.n:,}: `{last.reason}`"
    return f"{_SWEEP_END_LABELS[outcome]} at n={last.n:,}"


def write_time_table(grouped: dict, names: dict[str, str]) -> None:
    """Write the per-configuration largest-n-within-time table."""
    lines = [
        "| Solver | Largest n within the time budget | Sweep ended by |",
        "|---|---|---|",
    ]
    for (tool, config), rows in grouped.items():
        passing = [r.n for r in rows if passes_time(r)]
        largest = f"**{max(passing):,}**" if passing else "—"
        lines.append(f"| {_series_label(tool, config, names, markdown=True)} | {largest} | {_sweep_end(rows)} |")
    _write_generated("scaling_time.md", lines)


def write_memory_table(grouped: dict, fits: dict, names: dict[str, str]) -> None:
    """Write the per-configuration largest-n-within-memory table."""
    lines = [
        "| Solver | Largest n within memory | Determination |",
        "|---|---|---|",
    ]
    for tool, config in grouped:
        fit = fits.get(f"{tool}/{config}", {})
        largest = f"**{fit['max_n']:,}**" if fit.get("max_n") else "—"
        lines.append(
            f"| {_series_label(tool, config, names, markdown=True)} | {largest} | {fit.get('reason', 'no fit')} |"
        )
    _write_generated("scaling_memory.md", lines)


def _write_generated(name: str, lines: list[str]) -> None:
    """Write one generated markdown fragment."""
    path = GENERATED_DIR / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


# =================================================================================================
#  Main entrypoint
# =================================================================================================
def main() -> None:
    """Regenerate both charts and both tables from the tracked data files."""
    plt.style.use(STYLE_SHEET)
    names = _solver_names()
    grouped = _records_by_config(load_scaling_records(DATA_PATH))
    fits = json.loads(FIT_PATH.read_text(encoding="utf-8")) if FIT_PATH.exists() else {}
    render_time_chart(grouped, names)
    render_memory_chart(grouped, fits, names)
    write_time_table(grouped, names)
    write_memory_table(grouped, fits, names)


if __name__ == "__main__":
    main()
