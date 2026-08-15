"""Regenerate the benchmark-problem illustration images under docs/benchmarks/solver/images/.

For every built-in problem this renders three webp images from the current generators:
geometry (`problem_X.webp`), geometry with an example solution (`problem_X_with_solution.webp`,
DEFAULT preset over 10,000 iterations), and the nearest-neighbor separation distributions over a
range of problem sizes (`problem_X_separations.webp`).  All problems render their geometry at
n=200, where every problem in the suite is 2-dimensional.

Run:  uv run --group benchmarks --python 3.13 python scripts/generate_problem_images.py [PROBLEM ...]
"""

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde

from max_div.benchmark_problems import BenchmarkProblemFactory
from max_div.metrics import DiversityMetric
from max_div.solver import MaxDivSolverBuilder, SolverPreset, iterations

IMAGES_DIR = Path(__file__).parent.parent / "docs" / "benchmarks" / "solver" / "images"
GEOMETRY_N = 200
SEPARATION_N_VALUES = [200, 800, 3200, 12800]

GRAY = "#b0b0b0"
LIGHT_GRAY = "#cccccc"
BLUE = "#2233bb"
RED = "#ee1111"


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
        # cone from the origin tangent to the a=1 circle around (1,1) with radius r, plus its cap arc
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
    """Render the n=200 geometry scatter, optionally with a DEFAULT-preset example solution."""
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

    ax.legend(fontsize=15, loc="lower right" if name.startswith("C") else "upper right", framealpha=0.9)
    suffix = "_with_solution" if with_solution else ""
    _save_webp(fig, IMAGES_DIR / f"problem_{name}{suffix}.webp")


def render_separations(name: str) -> None:
    """Render nearest-neighbor L2-distance distributions (KDE on a log axis, with rugs) across sizes."""
    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    fig.subplots_adjust(top=0.87, left=0.09, right=0.96, bottom=0.10)
    fig.text(0.05, 0.945, f"Vector separations for problem {name}", fontsize=22, fontweight="bold", ha="left")
    fig.text(0.05, 0.905, "(full population; distances to nearest neighbor)", fontsize=17, ha="left")

    rug_base = -0.06
    for i, n in enumerate(SEPARATION_N_VALUES):
        problem = BenchmarkProblemFactory.construct_problem(
            name, n=n, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION
        )
        nn_dist = _nearest_neighbor_distances(problem.vectors)
        log_dist = np.log10(nn_dist[nn_dist > 0])
        kde = gaussian_kde(log_dist)
        grid = np.linspace(log_dist.min() - 0.5, log_dist.max() + 0.5, 400)
        density = kde(grid)
        color = f"C{i}"
        ax.plot(10**grid, density, color=color, linewidth=1.5)
        ax.fill_between(10**grid, density, color=color, alpha=0.25, label=f"$n$ = {n}")
        peak_x = 10 ** grid[np.argmax(density)]
        ax.text(peak_x, density.max() + 0.04, f"$n$ = {n}", fontsize=15, ha="center")
        rug_y = rug_base * (i + 1)
        ax.vlines(nn_dist, rug_y - 0.035, rug_y, color=color, linewidth=0.4, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$L_2$ distance", fontsize=15)
    ax.set_ylabel("Probability density (KDE)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.legend(title="Problem size", fontsize=14, title_fontsize=14, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save_webp(fig, IMAGES_DIR / f"problem_{name}_separations.webp")


def _nearest_neighbor_distances(vectors: np.ndarray) -> np.ndarray:
    """Exact nearest-neighbor L2 distance per vector, computed in row chunks to bound memory."""
    v = vectors.astype(np.float64)
    n = v.shape[0]
    result = np.empty(n)
    chunk = 1024
    sq_norms = (v * v).sum(axis=1)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d2 = sq_norms[start:stop, None] + sq_norms[None, :] - 2.0 * (v[start:stop] @ v.T)
        np.clip(d2, 0.0, None, out=d2)
        for row, idx in enumerate(range(start, stop)):
            d2[row, idx] = np.inf  # exclude self
        result[start:stop] = np.sqrt(d2.min(axis=1))
    return result


def main() -> None:
    """Regenerate images for the problems named on the command line (default: all)."""
    names = sys.argv[1:] or BenchmarkProblemFactory.get_all_benchmark_names()
    for name in names:
        render_geometry(name, with_solution=False)
        render_geometry(name, with_solution=True)
        render_separations(name)


if __name__ == "__main__":
    main()
