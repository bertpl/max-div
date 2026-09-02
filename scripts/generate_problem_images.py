"""Regenerate the benchmark-problem illustration images under docs/benchmarks/solver/images/.

For every built-in problem this renders two webp images from the current generators:
geometry (`problem_X.webp`) and geometry with an example solution (`problem_X_with_solution.webp`,
DEFAULT preset over 10,000 iterations), both at `GEOMETRY_N`, where every problem in the suite is
2-dimensional.  The separation-distribution images come from a separate generator.

Run:  uv run --group benchmarks --python 3.14 python scripts/generate_problem_images.py [PROBLEM ...]
"""

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from max_div.benchmark_problems import BenchmarkProblemFactory
from max_div.metrics import DiversityMetric
from max_div.solver import MaxDivSolverBuilder, SolverPreset, iterations

IMAGES_DIR = Path(__file__).parent.parent / "docs" / "benchmarks" / "solver" / "images"
GEOMETRY_N = 200

GRAY = "#b0b0b0"
LIGHT_GRAY = "#cccccc"
BLUE = "#2233bb"
RED = "#ee1111"

# Explicit per-problem axes and legend corner: without them, autoscale frames the data
# edge-to-edge and the legend has no empty corner to sit in (most visibly U4, where it lands on
# the cone). Each problem's legend goes in the corner its own data leaves empty, and each frame
# carries a margin around the data (U1's halo reaches ~1.3, U3 sits on a wider scale, U2 is a unit
# square) so nothing is clipped and the legend has room.
_XY_LIMS = {
    "U1": ((-0.4, 1.4), (-0.2, 1.4)),
    "U2": ((-0.1, 1.1), (-0.1, 1.25)),
    "U3": ((-3.7, 2.9), (-2.4, 4.4)),
    "U4": ((-0.05, 1.25), (-0.05, 1.25)),
    "C1": ((-0.1, 1.1), (-4.0, 5.5)),
    "C2": ((-0.1, 1.1), (-4.0, 5.5)),
    "C3": ((-5.0, 5.0), (-4.0, 6.0)),
    "C4": ((-5.0, 5.0), (-4.0, 6.0)),
}
_LEGEND_LOC = {
    "U1": "upper right",
    "U2": "upper right",
    "U3": "upper right",
    "U4": "upper left",
    "C1": "lower right",
    "C2": "lower right",
    "C3": "lower left",
    "C4": "lower left",
}


# =================================================================================================
#  Rendering helpers
# =================================================================================================
def _save_webp(fig: plt.Figure, path: Path) -> None:
    """Render the figure to quality-92 webp via PIL (matplotlib has no native webp writer)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200)
    plt.close(fig)
    buffer.seek(0)
    Image.open(buffer).convert("RGB").save(path, format="WEBP", lossless=False, quality=92)
    print(f"wrote {path.relative_to(IMAGES_DIR.parent.parent.parent.parent)}")


def _titled_figure(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    """Return a square figure with the suite's bold-title + regular-subtitle header."""
    fig, ax = plt.subplots(figsize=(10.0, 10.0))
    fig.subplots_adjust(top=0.88, left=0.11, right=0.95, bottom=0.09)
    fig.text(0.06, 0.955, title, fontsize=22, fontweight="bold", ha="left")
    fig.text(0.06, 0.92, subtitle, fontsize=17, ha="left")
    ax.set_xlabel("x1", fontsize=15)
    ax.set_ylabel("x2", fontsize=15)
    ax.tick_params(labelsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def _crosshair_gridlines(ax: plt.Axes) -> None:
    """Draw dashed gridlines at coordinates 0 and 1 on both axes."""
    for value in (0.0, 1.0):
        ax.axhline(value, color=LIGHT_GRAY, linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)
        ax.axvline(value, color=LIGHT_GRAY, linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)


def _draw_boundary(ax: plt.Axes, name: str) -> None:
    """Draw the light-gray population boundary appropriate for each problem's geometry."""
    if name in ("U1", "U2"):
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=LIGHT_GRAY, linewidth=1.5, label="Population boundary")
    if name == "U1":
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        for radius in (0.72, 0.82):
            ax.plot(0.5 + radius * np.cos(theta), 0.5 + radius * np.sin(theta), color=LIGHT_GRAY, linewidth=1.5)
    if name == "U3":
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), color=LIGHT_GRAY, linewidth=1.5, label=r"2$\sigma$ contour")
    if name == "U4":
        # draw the cone from the origin tangent to the a=1 circle around (1,1) with radius r, plus its cap arc
        r = 0.1 * np.sqrt(2.0)
        center_dist = np.sqrt(2.0)
        half_angle = np.arcsin(r / center_dist)
        tangent_len = np.sqrt(center_dist**2 - r**2)
        first = True
        tangent_points = []
        for phi in (np.pi / 4 - half_angle, np.pi / 4 + half_angle):
            tip = tangent_len * np.array([np.cos(phi), np.sin(phi)])
            ax.plot(
                [0, tip[0]],
                [0, tip[1]],
                color=LIGHT_GRAY,
                linewidth=1.5,
                label="Population boundary" if first else None,
            )
            tangent_points.append(tip)
            first = False
        center = np.array([1.0, 1.0])
        angles = [np.arctan2(*(p - center)[::-1]) for p in tangent_points]
        arc = np.linspace(angles[0], angles[1] + 2.0 * np.pi, 200)  # the long way, through the cone's far tip
        ax.plot(center[0] + r * np.cos(arc), center[1] + r * np.sin(arc), color=LIGHT_GRAY, linewidth=1.5)


def _draw_band_constraints(ax: plt.Axes, name: str, n: int) -> None:
    """Draw blue dash-dot constraint overlays: band boundaries for C1/C2, sign/band lines for C3/C4."""
    _, _, _k, m, _ = BenchmarkProblemFactory.get_problem_dimensions(name, n=n)
    if name in ("C1", "C2"):
        for i in range(m + 1):
            ax.axvline(
                i / m, color=BLUE, linestyle="-.", linewidth=1.5, label="Constraint group boundary" if i == 0 else None
            )
        label = "select\nexactly 5" if name == "C1" else "select\nat least 4"
        y_text = ax.get_ylim()[1]
        for i in range(m):
            x_mid = (i + 0.5) / m
            ax.annotate(
                "",
                xy=(i / m + 0.02, y_text * 0.82),
                xytext=((i + 1) / m - 0.02, y_text * 0.82),
                arrowprops={"arrowstyle": "<->", "color": BLUE, "linewidth": 1.5},
            )
            ax.text(x_mid, y_text * 0.86, label, color=BLUE, fontsize=13, ha="center", va="bottom")
    if name in ("C3", "C4"):
        # per-dimension sign constraints split each axis at 0
        ax.axvline(0.0, color=BLUE, linestyle="-.", linewidth=1.5, label="Constraint boundary")
        ax.axhline(0.0, color=BLUE, linestyle="-.", linewidth=1.5)
    if name == "C4":
        # the additional per-dimension band constraint covers [-1, +1]
        for value in (-1.0, 1.0):
            ax.axvline(value, color=BLUE, linestyle=":", linewidth=1.2)
            ax.axhline(value, color=BLUE, linestyle=":", linewidth=1.2)


# =================================================================================================
#  Image builders
# =================================================================================================
def render_geometry(name: str, with_solution: bool) -> None:
    """Render the GEOMETRY_N geometry scatter, optionally with a DEFAULT-preset example solution."""
    problem = BenchmarkProblemFactory.construct_problem(
        name, n=GEOMETRY_N, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION
    )
    _, _, k, m, _ = BenchmarkProblemFactory.get_problem_dimensions(name, n=GEOMETRY_N)
    subtitle = f"(select {k} of {GEOMETRY_N}"
    subtitle += f"; {m} constraint groups)" if m else ")"

    fig, ax = _titled_figure(f"Benchmark Problem {name}", subtitle)
    _crosshair_gridlines(ax)
    ax.scatter(problem.vectors[:, 0], problem.vectors[:, 1], s=45, color=GRAY, label="Population", zorder=2)
    _draw_boundary(ax, name)
    ax.set_xlim(*_XY_LIMS[name][0])
    ax.set_ylim(*_XY_LIMS[name][1])
    _draw_band_constraints(ax, name, GEOMETRY_N)

    if with_solution:
        solver = (
            MaxDivSolverBuilder(problem).with_preset(iterations(10_000), SolverPreset.DEFAULT).with_seed(42).build()
        )
        selected = solver.solve(verbosity=0).i_selected
        ax.scatter(
            problem.vectors[selected, 0],
            problem.vectors[selected, 1],
            s=140,
            color=RED,
            label="Example solution",
            zorder=3,
        )

    ax.legend(fontsize=15, loc=_LEGEND_LOC[name], framealpha=0.9)
    suffix = "_with_solution" if with_solution else ""
    _save_webp(fig, IMAGES_DIR / f"problem_{name}{suffix}.webp")


def main() -> None:
    """Regenerate images for the problems named on the command line (default: all)."""
    names = sys.argv[1:] or BenchmarkProblemFactory.get_all_benchmark_names()
    for name in names:
        render_geometry(name, with_solution=False)
        render_geometry(name, with_solution=True)


if __name__ == "__main__":
    main()
