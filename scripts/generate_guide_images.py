"""Generate the figures of the guide pages under `docs/guides/`.

Each figure plots one selection controlled by a parameter alpha: three cases as dot rows, and the
three separation metrics against alpha below them.

No solver is involved: the selections are given, and the figures show what each metric rewards.

Run with: ``uv run --group benchmarks ./scripts/generate_guide_images.py``.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from benchmarks.figures.style import REPO_ROOT, save_webp, use_docs_style

IMAGES_DIR = REPO_ROOT / "docs" / "guides" / "images"

# One color per metric, shared by every figure so the reader learns them once.
METRIC_COLORS = {"min separation": "#4C72B0", "mean separation": "#DD8452", "geometric-mean separation": "#55A868"}
CASE_LABELS = ("A", "B", "C")


# ==================================================================================================
#  Metrics of a one-dimensional selection
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


def main() -> None:
    """Render every guide figure."""
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


if __name__ == "__main__":
    main()
