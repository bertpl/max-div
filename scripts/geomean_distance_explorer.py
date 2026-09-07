"""Build the interactive example figure of `docs/guides/geomean_distance.md` as an HTML fragment.

The fragment is an inline SVG of the solved selection over a raster of the population, followed by a
caption. `docs/javascripts/geomean_explorer.js` adds the interaction: hovering an item highlights its
nearest neighbor under the geometric-mean distance and draws the level curve of that distance. Every
number the interaction needs is precomputed here and carried by `data-*` attributes, so the script
never recomputes a distance. Without JavaScript the fragment renders as a static figure.

This module depends on numpy only, so its tests run in the plain test environment; the figure
script under `scripts/` renders the population raster with Matplotlib and calls in here for the rest.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# --- data-space extent and pixel layout -----------
# The axes limits of the former static figure, kept so the interactive one looks the same.
X_MIN, X_MAX = -0.06, 1.02
Y_MIN, Y_MAX = -0.06, 1.12
VIEW_WIDTH = 624  # 6.5 inches at CSS resolution, the design width of the other guide figures
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 44, 8, 8, 36
SCALE = (VIEW_WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / (X_MAX - X_MIN)  # pixels per data unit
VIEW_HEIGHT = round(MARGIN_TOP + (Y_MAX - Y_MIN) * SCALE + MARGIN_BOTTOM)

# --- marks, in data units ------------------------
DOT_RADIUS = 0.008
RUG_NEAR, RUG_FAR = -0.018, -0.042  # a rug tick runs from RUG_NEAR to RUG_FAR beside its axis
TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

INK = "#222222"
POPULATION_COLOR = "#B0B0B0"
SELECTION_COLOR = "#EE1111"
NEIGHBOR_COLOR = "#4C72B0"

HINT = (
    "Hover over a red dot or one of its rug marks (tap on a touch screen) to see its nearest neighbor under the metric."
)


@dataclass(frozen=True)
class NearestNeighbors:
    """Per item: the nearest other item under the geometric-mean distance, and under the Euclidean one."""

    index: NDArray[np.intp]
    distance: NDArray[np.float64]
    dx: NDArray[np.float64]
    dy: NDArray[np.float64]
    euclidean_index: NDArray[np.intp]


def nearest_neighbors(x: NDArray[np.floating], y: NDArray[np.floating]) -> NearestNeighbors:
    """Return each item's nearest neighbor under the geometric-mean distance sqrt(|dx| |dy|) and the Euclidean one.

    Two items sharing a coordinate are at geometric-mean distance 0; that pair is then each other's
    nearest neighbor, and the caller draws the degenerate level curve.
    """
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    dx = np.abs(x64[:, None] - x64[None, :])
    dy = np.abs(y64[:, None] - y64[None, :])
    geomean = np.sqrt(dx * dy)
    euclidean = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(geomean, np.inf)
    np.fill_diagonal(euclidean, np.inf)
    index = geomean.argmin(axis=1)
    rows = np.arange(len(x64))
    return NearestNeighbors(
        index=index,
        distance=geomean[rows, index],
        dx=dx[rows, index],
        dy=dy[rows, index],
        euclidean_index=euclidean.argmin(axis=1),
    )


# ==================================================================================================
#  SVG pieces
# ==================================================================================================
def _px(x: float) -> float:
    """Return the pixel x of a data x."""
    return MARGIN_LEFT + (x - X_MIN) * SCALE


def _py(y: float) -> float:
    """Return the pixel y of a data y (pixel y grows downward)."""
    return MARGIN_TOP + (Y_MAX - y) * SCALE


def _axes() -> list[str]:
    """Return the two spines with their ticks and tick labels, in pixel space."""
    left, bottom, top, right = _px(X_MIN), _py(Y_MIN), _py(Y_MAX), _px(X_MAX)
    parts = [
        f'<path class="gmx-spine" d="M{left:.1f},{top:.1f}V{bottom:.1f}H{right:.1f}"/>',
    ]
    for tick in TICKS:
        x, y = _px(tick), _py(tick)
        parts.append(f'<line class="gmx-tick" x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 4:.1f}"/>')
        parts.append(f'<text class="gmx-label" x="{x:.1f}" y="{bottom + 17:.1f}" text-anchor="middle">{tick:g}</text>')
        parts.append(f'<line class="gmx-tick" x1="{left:.1f}" y1="{y:.1f}" x2="{left - 4:.1f}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="gmx-label" x="{left - 7:.1f}" y="{y:.1f}" text-anchor="end" dominant-baseline="middle">'
            f"{tick:g}</text>"
        )
    return parts


def _legend(n: int, k: int) -> list[str]:
    """Return the legend box in the top-right corner, in pixel space."""
    width, row, pad = 262, 18, 8
    x0 = _px(X_MAX) - 6 - width
    y0 = _py(Y_MAX) + 6
    entries = (
        (f'<circle cx="{x0 + 14}" cy="{{y}}" r="1.5" fill="{POPULATION_COLOR}"/>', f"population (n = {n:,})"),
        (f'<circle cx="{x0 + 14}" cy="{{y}}" r="4.5" fill="{SELECTION_COLOR}"/>', f"selection (k = {k})"),
        (
            f'<circle cx="{x0 + 14}" cy="{{y}}" r="4.5" fill="{NEIGHBOR_COLOR}"/>',
            "nearest neighbor, geometric-mean distance",
        ),
        (
            f'<circle cx="{x0 + 14}" cy="{{y}}" r="5.5" fill="none" stroke="{INK}" stroke-width="1.2"'
            ' stroke-dasharray="2 2"/>',
            "nearest neighbor, Euclidean distance",
        ),
    )
    height = pad * 2 + row * len(entries)
    parts = [f'<rect class="gmx-legend" x="{x0}" y="{y0}" width="{width}" height="{height}"/>']
    for i, (mark, label) in enumerate(entries):
        y = y0 + pad + row * i + row / 2
        parts.append(mark.format(y=f"{y:.1f}"))
        parts.append(f'<text class="gmx-label" x="{x0 + 26}" y="{y:.1f}" dominant-baseline="middle">{label}</text>')
    return parts


def _data_group(x: NDArray[np.floating], y: NDArray[np.floating], neighbors: NearestNeighbors) -> list[str]:
    """Return the group in data coordinates: the hover layer, the rug ticks with their hit areas, and the dots."""
    parts = [
        f'<g class="gmx-data" transform="translate({_px(0.0):.2f},{_py(0.0):.2f}) scale({SCALE:.3f},{-SCALE:.3f})">',
        '<g class="gmx-hover" clip-path="url(#gmx-square)"></g>',
    ]
    for cls in ("gmx-rug", "gmx-hit"):
        for i, (xi, yi) in enumerate(zip(x, y)):
            parts.append(
                f'<line class="{cls} gmx-rug-x" data-i="{i}" x1="{xi:.5f}" y1="{RUG_NEAR}" x2="{xi:.5f}" y2="{RUG_FAR}"'
                ' vector-effect="non-scaling-stroke"/>'
            )
            parts.append(
                f'<line class="{cls} gmx-rug-y" data-i="{i}" x1="{RUG_NEAR}" y1="{yi:.5f}" x2="{RUG_FAR}" y2="{yi:.5f}"'
                ' vector-effect="non-scaling-stroke"/>'
            )
    for i, (xi, yi) in enumerate(zip(x, y)):
        parts.append(
            f'<circle class="gmx-dot" data-i="{i}" data-nn="{neighbors.index[i]}"'
            f' data-nne="{neighbors.euclidean_index[i]}"'
            f' data-d="{neighbors.distance[i]:.5f}" data-dx="{neighbors.dx[i]:.5f}" data-dy="{neighbors.dy[i]:.5f}"'
            f' cx="{xi:.5f}" cy="{yi:.5f}" r="{DOT_RADIUS}" vector-effect="non-scaling-stroke"'
            f' tabindex="0" role="button" aria-label="item {i}"/>'
        )
    parts.append("</g>")
    return parts


def explorer_fragment(
    x: NDArray[np.floating], y: NDArray[np.floating], n: int, k: int, population_image: str, description: str
) -> str:
    """Return the HTML fragment: a `<div>` holding the SVG and its caption, ending in a newline.

    Args:
        x, y: Coordinates of the k selected items, as the solver saw them.
        n: Population size, for the legend.
        k: Selection size, for the legend and the 1/sqrt(k) reference level.
        population_image: URL of the population raster, relative to the page that includes the fragment.
        description: Alternative text of the figure.
    """
    neighbors = nearest_neighbors(x, y)
    square_x, square_y = _px(0.0), _py(1.0)
    lines = [
        '<div class="gmx-figure">',
        f'<svg class="gmx" viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" xmlns="http://www.w3.org/2000/svg" role="img"'
        f' data-k="{k}" data-ref="{1.0 / np.sqrt(k):.5f}">',
        "<title>Selection under the geometric-mean distance</title>",
        f"<desc>{description}</desc>",
        '<defs><clipPath id="gmx-square" clipPathUnits="userSpaceOnUse">'
        '<rect x="0" y="0" width="1" height="1"/></clipPath></defs>',
        f'<rect x="0" y="0" width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}" fill="#ffffff"/>',
        f'<image href="{population_image}" x="{square_x:.2f}" y="{square_y:.2f}"'
        f' width="{SCALE:.2f}" height="{SCALE:.2f}" preserveAspectRatio="none"/>',
        *_axes(),
        *_legend(n, k),
        *_data_group(x, y, neighbors),
        "</svg>",
        f'<p class="gmx-caption">{HINT}</p>',
        "</div>",
    ]
    return "\n".join(lines) + "\n"
