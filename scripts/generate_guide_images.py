"""Generate the figures of the guide pages under `docs/guides/`.

The figures of `geomean_separation.md` plot given selections controlled by a parameter alpha: three
cases as dot rows, and the three separation metrics against alpha below them; no solver is involved.
The figures of `geomean_distance.md` plot the metric's level curves and one solved selection; only
that one runs the solver. That selection is emitted as an interactive figure (an HTML fragment over a
raster of the population) plus its separations table, and cached as JSON so that `--reuse-solution`
re-renders it without the solve.

Run with: ``uv run --group benchmarks ./scripts/generate_guide_images.py [--reuse-solution]``.
"""

import argparse
import json
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from geomean_distance_explorer import POPULATION_COLOR, explorer_fragment
from numpy.typing import NDArray

from benchmarks.figures.style import REPO_ROOT, save_webp, use_docs_style
from max_div.metrics import DistanceMetric, DiversityMetric
from max_div.problem import MaxDivProblem
from max_div.solver import ParallelMaxDivSolverBuilder, seconds

GENERATED_DIR = REPO_ROOT / "generated"
IMAGES_DIR = REPO_ROOT / "docs" / "guides" / "images"
SOLUTION_PATH = GENERATED_DIR / "geomean_distance_example_solution.json"

# One color per metric, shared by every figure so the reader learns them once.
METRIC_COLORS = {"min separation": "#4C72B0", "mean separation": "#DD8452", "geometric-mean separation": "#55A868"}
CASE_LABELS = ("A", "B", "C")


# ==================================================================================================
#  Metrics of a selection
# ==================================================================================================
def separations(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return each item's distance to its nearest other item, for sorted one-dimensional positions."""
    gaps = np.diff(positions)
    left = np.concatenate(([np.inf], gaps))
    right = np.concatenate((gaps, [np.inf]))
    return np.minimum(left, right)


def metrics(positions: NDArray[np.float64]) -> dict[str, float]:
    """Return the three separation-based diversity metrics of a one-dimensional selection."""
    sep = separations(np.sort(positions))
    return {
        "min separation": float(sep.min()),
        "mean separation": float(sep.mean()),
        "geometric-mean separation": float(np.exp(np.mean(np.log(sep)))) if np.all(sep > 0) else 0.0,
    }


def geomean_separation_2d(points: NDArray[np.float64]) -> float:
    """Return the geometric mean over points of the Euclidean distance to each point's nearest other point."""
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(distances, np.inf)
    return float(np.exp(np.mean(np.log(distances.min(axis=1)))))


# ==================================================================================================
#  Rendering
# ==================================================================================================
def render_example(
    name: str,
    layout: Callable[[float], NDArray[np.float64]],
    case_alphas: tuple[float, float, float],
    alpha_range: tuple[float, float],
    axis_range: tuple[float, float],
    legend: dict | None = None,
    diversity_max: float | None = None,
    highlight: tuple[int, ...] = (),
    dot_size: float = 36,
    position_marks: tuple[float, ...] | None = None,
) -> None:
    """Render one figure: the three cases as dot rows, and the metrics against alpha below them.

    Args:
        name: Image file stem under `IMAGES_DIR`.
        layout: Maps alpha to the item positions of the selection.
        alpha_range: The alpha interval the metric curves cover.
        axis_range: The position interval every dot row shows, shared so the cases compare.
        legend: Keyword arguments placing the metric legend where the curves leave room.
        diversity_max: Top of the diversity axis; autoscaled when omitted.
        highlight: Indices of the items alpha acts on, drawn as hollow rings.
        dot_size: Marker area of the items; smaller for selections with many items.
        position_marks: Positions marked by a labeled vertical across the rows; the whole numbers in range when omitted.
    """
    use_docs_style()
    fig, (ax_cases, ax_metrics) = plt.subplots(2, 1, figsize=(8.0, 5.6), height_ratios=(1.3, 2.0))
    fig.subplots_adjust(hspace=0.45, left=0.09, right=0.91, top=0.95, bottom=0.09)

    # --- the cases as dot rows ------------------
    for row, (label, alpha) in enumerate(zip(CASE_LABELS, case_alphas)):
        y = len(CASE_LABELS) - 1 - row
        positions = layout(alpha)
        ax_cases.hlines(y, *axis_range, color="#D8D8D8", linewidth=1.0, zorder=1)
        is_highlighted = np.isin(np.arange(positions.size), highlight)
        ax_cases.scatter(
            positions[~is_highlighted], np.full((~is_highlighted).sum(), y), s=dot_size, c="#222222", zorder=2
        )
        ax_cases.scatter(
            positions[is_highlighted],
            np.full(is_highlighted.sum(), y),
            s=60,
            facecolors="white",
            edgecolors="#222222",
            linewidths=1.8,
            zorder=3,
        )
        pad = 0.02 * (axis_range[1] - axis_range[0])
        ax_cases.text(axis_range[0] - pad, y, f"Case {label}", ha="right", va="center")
        ax_cases.text(axis_range[1] + pad, y, f"\u03b1 = {alpha:g}", ha="left", va="center")  # alpha
    # Faint labeled verticals at position_marks let a position be read across the rows.
    y_top = len(CASE_LABELS) - 0.4
    if position_marks is None:
        position_marks = tuple(np.arange(np.ceil(axis_range[0]), axis_range[1] + 1e-9) + 0.0)  # + 0.0 turns a -0 into 0
    for tick in position_marks:
        ax_cases.vlines(tick, -0.6, y_top, color="#D8D8D8", linewidth=0.8, linestyle="--", zorder=0)
        label = f"{tick:.1f}" if float(tick).is_integer() else f"{tick:g}"
        ax_cases.text(tick, y_top, label, ha="center", va="bottom", color="#888888", fontsize="small")
    ax_cases.set_xlim(*axis_range)
    ax_cases.set_ylim(-0.6, y_top)
    ax_cases.set_yticks([])
    ax_cases.set_xticks([])  # the labeled verticals are the position scale
    for side in ("left", "right", "top", "bottom"):
        ax_cases.spines[side].set_visible(False)

    # --- the metrics against alpha --------------
    alphas = np.linspace(*alpha_range, 400)
    curves = {metric: np.array([metrics(layout(a))[metric] for a in alphas]) for metric in METRIC_COLORS}
    for metric, color in METRIC_COLORS.items():
        ax_metrics.plot(alphas, curves[metric], color=color, linewidth=1.8, label=metric)
    if diversity_max is not None:
        ax_metrics.set_ylim(top=diversity_max)
    for label, alpha in zip(CASE_LABELS, case_alphas):
        ax_metrics.axvline(alpha, color="#555555", linestyle=":", linewidth=1.2)
        ax_metrics.text(alpha, ax_metrics.get_ylim()[1], f" {label}", ha="left", va="top", color="#555555")
    ax_metrics.set_xlim(*alpha_range)
    ax_metrics.set_xlabel("\u03b1")  # alpha
    ax_metrics.set_ylabel("diversity")
    ax_metrics.legend(**(legend or {"loc": "best"}))

    save_webp(fig, IMAGES_DIR / f"{name}.webp")


# ==================================================================================================
#  Examples
# ==================================================================================================
def layout_closest_pair_fixed(alpha: float) -> NDArray[np.float64]:
    """Return items with the first two at 0.0 and 0.1 and every further item alpha after the previous."""
    return np.concatenate(([0.0, 0.1], 0.1 + alpha * np.arange(1, 10)))


def layout_near_duplicate(alpha: float) -> NDArray[np.float64]:
    """Return items uniform over [0, 1], except the second, which sits at alpha."""
    positions = np.linspace(0.0, 1.0, 11)
    positions[1] = alpha
    return positions


def layout_power_spacing(alpha: float) -> NDArray[np.float64]:
    """Return items power-spaced over [0, 1]: uniform at alpha = 1, crowding toward 0 below it and toward 1 above it."""
    return (np.arange(51) / 50.0) ** ((2.0 - alpha) / alpha)


def layout_constrained_group(alpha: float) -> NDArray[np.float64]:
    """Return 26 items pinned uniformly over [-0.25, 0] plus free items over (0, 1] crowding toward 0 for alpha > 0."""
    constrained = -0.25 + np.arange(26) / 100.0
    t = np.arange(26, 51) / 25.0 - 1.0
    free = (1.0 - alpha) * t + alpha * t**2
    return np.concatenate((constrained, free))


# ==================================================================================================
#  Geometric-mean distance: level curves and a solved example
# ==================================================================================================
SELECTION_COLOR = "#ee1111"


def render_geomean_distance_levels(name: str, k: int) -> None:
    """Plot the level curves of the geometric-mean distance from the origin, over all four quadrants.

    The curve at distance 1/sqrt(k) is highlighted; `docs/guides/geomean_distance.md`, section II,
    explains why. Two fainter curves show the family around the highlighted curve.

    Args:
        name: Image file stem under `IMAGES_DIR`.
        k: Sets the highlighted level, 1/sqrt(k).
    """
    use_docs_style()
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.1)
    grid = np.linspace(-1.0, 1.0, 1201)
    dx, dy = np.meshgrid(grid, grid)
    distance = np.sqrt(np.abs(dx * dy))
    reference = 1.0 / np.sqrt(k)
    levels = (reference / 2, reference, 0.5)
    contours = ax.contour(
        dx, dy, distance, levels=levels, colors=("#BBBBBB", SELECTION_COLOR, "#BBBBBB"), linewidths=(1.0, 2.0, 1.0)
    )
    # label each curve where it crosses Δx = 0.45 in the first quadrant, clear of the marked points
    ax.clabel(
        contours, fmt=lambda v: f"d = {v:g}", fontsize="small", manual=[(0.45, level**2 / 0.45) for level in levels]
    )
    ax.axhline(0.0, color="#D8D8D8", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#D8D8D8", linewidth=0.8, zorder=0)
    for point, label, offset in (
        ((reference, reference), "(1/\u221ak, 1/\u221ak)", (8, 8)),
        ((1.0, 1.0 / k), "(1, 1/k)", (-30, 12)),
    ):
        ax.plot(*point, "o", color="#222222", markersize=6, zorder=3)
        ax.annotate(label, point, textcoords="offset points", xytext=offset)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Δx")
    ax.set_ylabel("Δy")
    ax.set_aspect("equal")
    save_webp(fig, IMAGES_DIR / f"{name}.webp")


def population(n: int, seed: int) -> NDArray[np.float32]:
    """Return the n evenly spaced values in [0, 1] as x, paired with a seeded random permutation of them as y.

    Pairing one evenly spaced grid with a permutation of itself gives every marginal a minimum spacing
    of 1 / (n - 1), which a random sample lacks.
    """
    rng = np.random.default_rng(seed)
    values = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.column_stack((values, rng.permutation(values)))


def solve_geomean_distance_example(
    vectors: NDArray[np.float32], k: int, budget_sec: float, n_workers: int, seed: int
) -> NDArray[np.intp]:
    """Return the selected indices: geometric-mean separation under the geometric-mean distance, end-to-end budget."""
    problem = MaxDivProblem.new(
        vectors=vectors,
        k=k,
        distance_metric=DistanceMetric.geometric_mean(),
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
    )
    solver = (
        ParallelMaxDivSolverBuilder(problem)
        .with_seed(seed)
        .with_workers(seconds(budget_sec), n_workers)
        .with_end_to_end_budget()
        .build()
    )
    return solver.solve(verbosity=0).i_selected


def load_or_solve(
    n: int, k: int, budget_sec: float, n_workers: int, seed: int, reuse_solution: bool
) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
    """Return the population and the selected indices, from the JSON cache when asked, else from a fresh solve.

    A fresh solve rewrites the cache. The cache holds the selected indices only: the population is
    rebuilt from n and the seed, so the dots and the raster always come from the same coordinates.
    """
    if reuse_solution:
        cached = json.loads(SOLUTION_PATH.read_text(encoding="utf-8"))
        settings = {"n": n, "k": k, "budget_sec": budget_sec, "n_workers": n_workers, "seed": seed}
        if {key: cached[key] for key in settings} != settings:
            raise ValueError(f"{SOLUTION_PATH} was solved with other settings than {settings}")
        return population(n, seed), np.asarray(cached["i_selected"], dtype=np.intp)
    vectors = population(n, seed)
    selected = solve_geomean_distance_example(vectors, k, budget_sec, n_workers, seed)
    record = {
        "n": n,
        "k": k,
        "budget_sec": budget_sec,
        "n_workers": n_workers,
        "seed": seed,
        "i_selected": [int(i) for i in selected],
    }
    SOLUTION_PATH.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {SOLUTION_PATH.relative_to(REPO_ROOT)}")
    return vectors, selected


def render_geomean_distance_population(name: str, vectors: NDArray[np.float32], pixels: int = 1000) -> None:
    """Render the population alone as a square lossless raster covering exactly the unit square.

    The interactive figure lays its SVG marks over this image, so the image carries no axes or margins:
    its edges are the square's edges.
    """
    use_docs_style()
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        dpi = plt.rcParams["savefig.dpi"]
        fig = plt.figure(figsize=(pixels / dpi, pixels / dpi))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        ax.scatter(vectors[:, 0], vectors[:, 1], s=0.7, color=POPULATION_COLOR, linewidths=0)
        save_webp(fig, IMAGES_DIR / f"{name}.webp", lossless=True)


def write_geomean_distance_example_separations(name: str, selection: NDArray[np.float64], k: int) -> None:
    """Write the selection's achieved separations next to the scaling targets 1/sqrt(k) and 1/k as a table fragment.

    `docs/guides/geomean_distance.md` includes the fragment, so the numbers under the figure come from the
    same solve as the figure.
    """
    rows = (
        ("2D, Euclidean distance", geomean_separation_2d(selection), r"$1/\sqrt{k}$", 1.0 / np.sqrt(k)),
        ("$x$ marginal", metrics(selection[:, 0])["geometric-mean separation"], "$1/k$", 1.0 / k),
        ("$y$ marginal", metrics(selection[:, 1])["geometric-mean separation"], "$1/k$", 1.0 / k),
    )
    lines = ["| geometric-mean separation | achieved | target | target value |", "|---|---|---|---|"]
    lines += [f"| {label} | {achieved:.4f} | {formula} | {target:.4f} |" for label, achieved, formula, target in rows]
    path = GENERATED_DIR / f"{name}_separations.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def render_geomean_distance_example(
    name: str, n: int, k: int, budget_sec: float, n_workers: int, seed: int, reuse_solution: bool
) -> None:
    """Produce the example: the population raster, the interactive figure fragment and the separations table.

    The selection maximizes geometric-mean separation under the geometric-mean distance on the
    `population` of n items.

    Args:
        name: File stem of the raster under `IMAGES_DIR` and of the fragments under `GENERATED_DIR`.
        seed: Seeds both the pairing and the solver.
        reuse_solution: Read the cached selection instead of solving again.
    """
    vectors, selected = load_or_solve(n, k, budget_sec, n_workers, seed, reuse_solution)
    selection = vectors[selected].astype(np.float64)
    write_geomean_distance_example_separations(name, selection, k)
    render_geomean_distance_population(f"{name}_population", vectors)
    fragment = explorer_fragment(
        selection[:, 0],
        selection[:, 1],
        n,
        k,
        population_image=f"../images/{name}_population.webp",
        description=(
            "Ten thousand gray points in the unit square with the hundred selected ones in red, and the "
            "selection's x and y values as rug marks along the bottom and left edges"
        ),
    )
    path = GENERATED_DIR / f"{name}_figure.html"
    path.write_text(fragment, encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    """Render every guide figure; `--reuse-solution` skips the example's solve and reads its cache."""
    parser = argparse.ArgumentParser(description="Regenerate the guide figures.")
    parser.add_argument("--reuse-solution", action="store_true", help="read the cached example selection")
    args = parser.parse_args()
    render_example(
        "geomean_separation_I1",
        layout_closest_pair_fixed,
        case_alphas=(0.1, 0.2, 0.3),
        alpha_range=(0.05, 0.35),
        axis_range=(-0.1, 3.1),
        legend={"loc": "upper left", "bbox_to_anchor": (0.19, 1.0)},
        highlight=(0, 1),
    )
    render_example(
        "geomean_separation_I2",
        layout_near_duplicate,
        case_alphas=(0.001, 0.01, 0.1),
        alpha_range=(0.0, 0.2),
        axis_range=(-0.1, 1.1),
        legend={"loc": "lower center"},
        diversity_max=0.12,
        highlight=(1,),
    )
    render_example(
        "geomean_separation_I3",
        layout_power_spacing,
        case_alphas=(0.5, 1.0, 1.75),
        alpha_range=(0.01, 1.99),
        axis_range=(-0.1, 1.1),
        legend={"loc": "upper left", "bbox_to_anchor": (0.01, 0.92)},
        diversity_max=0.022,
        dot_size=14,
    )
    render_example(
        "geomean_separation_II",
        layout_constrained_group,
        case_alphas=(-0.5, 0.0, 0.75),
        alpha_range=(-1.0, 1.0),
        axis_range=(-0.35, 1.1),
        dot_size=14,
        position_marks=(-0.25, 0.0, 1.0),
    )
    render_geomean_distance_levels("geomean_distance_levels", k=25)
    render_geomean_distance_example(
        "geomean_distance_example",
        n=10_000,
        k=100,
        budget_sec=60.0,
        n_workers=16,
        seed=42,
        reuse_solution=args.reuse_solution,
    )


if __name__ == "__main__":
    main()
