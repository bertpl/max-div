"""Build the interactive example figure of `docs/guides/geomean_distance.md` as an HTML fragment.

The fragment is an inline SVG of the solved selection over a raster of the population, followed by a
caption. `docs/javascripts/geomean_distance_explorer.js` adds the interaction. Every number the
interaction needs is precomputed here and carried by `data-*` attributes, so the JavaScript never
recomputes a distance. Without JavaScript the fragment renders as a static figure.

This module depends on numpy only, so its tests run without the `benchmarks` dependency group;
`generate_guide_images.py` renders the population raster with Matplotlib and calls `explorer_fragment`
for the rest.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# The axes extend past the unit square: room for the rug ticks below and left of it, and for the
# legend above it.
X_MIN, X_MAX = -0.06, 1.02
Y_MIN, Y_MAX = -0.06, 1.15
VIEW_WIDTH = 624  # 6.5 inches at CSS 96 px per inch, the scale the raster guide figures are shown at
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 44, 8, 8, 36
SCALE = (VIEW_WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / (X_MAX - X_MIN)  # pixels per data unit
VIEW_HEIGHT = round(MARGIN_TOP + (Y_MAX - Y_MIN) * SCALE + MARGIN_BOTTOM)

# The marks are placed and sized in data units, inside the scaled `gmx-data` group.
DOT_RADIUS = 0.008
RUG_NEAR, RUG_FAR = -0.018, -0.042  # a rug tick runs from RUG_NEAR to RUG_FAR beside its axis
TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# A reference neighbor is marked by a dashed ring around its dot, with an "x" or "+" inscribed for
# the marginal neighbors. The legend draws the ring in pixels; the JavaScript draws it in data units
# around the dot, from `RING_RADIUS_FACTOR` and `GLYPH_REACH_FACTOR`, carried as `data-ring` and
# `data-reach`.
RING_RADIUS_FACTOR = 1.7  # the ring radius is this many dot radii
GLYPH_REACH_FACTOR = 0.7  # a glyph stroke reaches this far from the center, in ring radii, so the "x" touches the ring
LEGEND_RING_RADIUS = 5.5

FOREGROUND_COLOR = "#222222"
POPULATION_COLOR = "#B0B0B0"
SELECTION_COLOR = "#EE1111"
NEIGHBOR_COLOR = "#4C72B0"

HINT = (
    "Hover over a red dot or one of its rug marks (tap on a touch screen) to see its nearest neighbor under the"
    " metric and the metric's level curve through that neighbor."
)


@dataclass(frozen=True)
class NearestNeighbors:
    """Parallel arrays record, per item, its nearest other item under each distance the figure marks.

    - `index`, `distance`, `dx`, `dy`: the neighbor under the geometric-mean distance;
    - `euclidean_index`: the neighbor under the Euclidean distance;
    - `x_index`, `y_index`: the neighbor along each marginal.
    """

    index: NDArray[np.intp]
    distance: NDArray[np.float64]
    dx: NDArray[np.float64]
    dy: NDArray[np.float64]
    euclidean_index: NDArray[np.intp]
    x_index: NDArray[np.intp]
    y_index: NDArray[np.intp]


def nearest_neighbors(x: NDArray[np.floating], y: NDArray[np.floating]) -> NearestNeighbors:
    """Return each item's nearest neighbor under the geometric-mean distance, the Euclidean one, and each marginal.

    Two items sharing a coordinate are at geometric-mean distance 0; that pair is then each other's
    nearest neighbor, and `geomean_distance_explorer.js` draws the degenerate level curve.
    """
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    dx = np.abs(x64[:, None] - x64[None, :])
    dy = np.abs(y64[:, None] - y64[None, :])
    geomean = np.sqrt(dx * dy)
    euclidean = np.sqrt(dx * dx + dy * dy)
    for matrix in (geomean, euclidean, dx, dy):
        np.fill_diagonal(matrix, np.inf)
    index = geomean.argmin(axis=1)
    rows = np.arange(len(x64))
    return NearestNeighbors(
        index=index,
        distance=geomean[rows, index],
        dx=dx[rows, index],
        dy=dy[rows, index],
        euclidean_index=euclidean.argmin(axis=1),
        x_index=dx.argmin(axis=1),
        y_index=dy.argmin(axis=1),
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


def _glyph_paths(x: float, y: float, ring_radius: float, glyph: str) -> str:
    """Return the two strokes of an "x" or "+" inscribed in a ring of `ring_radius` at (x, y), or nothing for `""`."""
    if glyph == "":
        return ""
    reach = GLYPH_REACH_FACTOR * ring_radius
    if glyph == "x":
        d = (
            f"M{x - reach:.1f},{y - reach:.1f}L{x + reach:.1f},{y + reach:.1f}"
            f"M{x - reach:.1f},{y + reach:.1f}L{x + reach:.1f},{y - reach:.1f}"
        )
    else:
        d = f"M{x - reach:.1f},{y:.1f}L{x + reach:.1f},{y:.1f}M{x:.1f},{y - reach:.1f}L{x:.1f},{y + reach:.1f}"
    return f'<path class="gmx-glyph" d="{d}"/>'


def _legend_mark(x: float, y: float, glyph: str) -> str:
    """Return a legend mark: the dashed ring drawn around a reference neighbor, with its inscribed glyph."""
    ring = f'<circle class="gmx-mark" cx="{x:.1f}" cy="{y:.1f}" r="{LEGEND_RING_RADIUS}"/>'
    return ring + _glyph_paths(x, y, LEGEND_RING_RADIUS, glyph)


def _legend(n: int, k: int) -> list[str]:
    """Return the legend box in the top-right corner, in pixel space.

    The left column names what is always drawn; the right column names the reference neighbors the
    interaction marks around the picked item.
    """
    column_widths, row, pad = (260, 224), 18, 8
    width = sum(column_widths) + pad
    x0 = _px(X_MAX) - 6 - width
    y0 = _py(Y_MAX) + 6
    columns = (
        (
            (
                lambda x, y: f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="{POPULATION_COLOR}"/>',
                f"population (n = {n:,})",
            ),
            (
                lambda x, y: f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{SELECTION_COLOR}"/>',
                f"selection (k = {k})",
            ),
            (
                lambda x, y: f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{NEIGHBOR_COLOR}"/>',
                "nearest neighbor, geometric-mean distance",
            ),
        ),
        (
            (lambda x, y: _legend_mark(x, y, ""), "nearest neighbor, Euclidean distance"),
            (lambda x, y: _legend_mark(x, y, "x"), "nearest neighbor, x marginal"),
            (lambda x, y: _legend_mark(x, y, "+"), "nearest neighbor, y marginal"),
        ),
    )
    height = pad * 2 + row * max(len(column) for column in columns)
    parts = [f'<rect class="gmx-legend" x="{x0}" y="{y0}" width="{width}" height="{height}"/>']
    x = x0
    for column, column_width in zip(columns, column_widths):
        for i, (mark, label) in enumerate(column):
            y = y0 + pad + row * i + row / 2
            parts.append(mark(x + 14, y))
            parts.append(
                f'<text class="gmx-label" x="{x + 26:.1f}" y="{y:.1f}" dominant-baseline="middle">{label}</text>'
            )
        x += column_width
    return parts


def _data_group(x: NDArray[np.floating], y: NDArray[np.floating], neighbors: NearestNeighbors) -> list[str]:
    """Return the group in data coordinates: the hover layer, the rug ticks and their hit areas, the dots, the marks.

    Children are listed in paint order: the marks layer comes last so rings draw over the dots.
    """
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
            f' data-nne="{neighbors.euclidean_index[i]}" data-nnx="{neighbors.x_index[i]}"'
            f' data-nny="{neighbors.y_index[i]}"'
            f' data-d="{neighbors.distance[i]:.5f}" data-dx="{neighbors.dx[i]:.5f}" data-dy="{neighbors.dy[i]:.5f}"'
            f' cx="{xi:.5f}" cy="{yi:.5f}" r="{DOT_RADIUS}" vector-effect="non-scaling-stroke"'
            f' tabindex="0" role="button" aria-label="item {i}"/>'
        )
    parts.append('<g class="gmx-marks"></g>')
    parts.append("</g>")
    return parts


def explorer_fragment(
    x: NDArray[np.floating], y: NDArray[np.floating], n: int, k: int, population_image: str, description: str
) -> str:
    """Return the HTML fragment: a `<div>` holding the SVG and its caption, ending in a newline.

    Args:
        x, y: Coordinates of the k selected items, in the solver's input coordinates.
        n: Population size, for the legend.
        k: Selection size, for the legend and the 1/sqrt(k) value in the caption.
        population_image: URL of the population raster, relative to the page that includes the fragment.
        description: Alternative text of the figure.
    """
    neighbors = nearest_neighbors(x, y)
    square_x, square_y = _px(0.0), _py(1.0)
    lines = [
        '<div class="gmx-figure">',
        f'<svg class="gmx" viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" xmlns="http://www.w3.org/2000/svg" role="img"'
        f' data-ref="{1.0 / np.sqrt(k):.5f}" data-ring="{RING_RADIUS_FACTOR}" data-reach="{GLYPH_REACH_FACTOR}">',
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
