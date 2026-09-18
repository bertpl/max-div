"""Build the interactive figures of `docs/guides/uniform_sampling.md` as HTML fragments.

Each fragment is an inline SVG of one experiment's selection over a raster of the population, followed
by a caption. `docs/javascripts/uniform_sampling_explorer.js` adds the interaction. Every number the
interaction needs is precomputed here and carried by `data-*` attributes, so the JavaScript never
recomputes a distance. Without JavaScript a fragment renders as a static figure.

An experiment maximizes a diversity objective over one or several of the distances in `DISTANCES`.
Per item, the fragment records the nearest other item under every distance the objective uses, and
under the three reference distances (L2, x, y) the interaction always marks.

This module depends on numpy only, so its tests run without the `benchmarks` dependency group;
`generate_guide_images.py` renders the population raster with Matplotlib and calls `explorer_fragment`
for the rest.
"""

import json
from collections.abc import Callable
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

# The marks are placed and sized in data units, inside the scaled `usx-data` group.
DOT_RADIUS = 0.008
RUG_NEAR, RUG_FAR = -0.018, -0.042  # a rug tick runs from RUG_NEAR to RUG_FAR beside its axis
TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# A reference neighbor is marked by a dashed ring around its dot with a glyph inscribed: a central dot
# for the L2 neighbor, a vertical stroke for the x neighbor, a horizontal stroke for the y neighbor.
# The legend draws the ring in pixels; the JavaScript draws it in data units around the dot, from
# `RING_RADIUS_FACTOR` and `GLYPH_REACH_FACTOR`, carried as `data-ring` and `data-reach`.
RING_RADIUS_FACTOR = 1.7  # the ring radius is this many dot radii
GLYPH_REACH_FACTOR = 0.7  # a stroke glyph reaches this far from the center, in ring radii
CENTER_DOT_FACTOR = 0.3  # the central-dot glyph has this radius, in ring radii
LEGEND_RING_RADIUS = 5.5

FOREGROUND_COLOR = "#222222"
POPULATION_COLOR = "#B0B0B0"
SELECTION_COLOR = "#EE1111"
NEIGHBOR_COLOR = "#4C72B0"

HINT = (
    "Hover over a red dot or one of its rug marks (tap on a touch screen) to see its nearest neighbors and the"
    " level curve of each objective distance through the neighbor under that distance."
)


# ==================================================================================================
#  Distances
# ==================================================================================================
@dataclass(frozen=True)
class Distance:
    """A distance the figure can mark: its key in the `data-*` attributes, its label, and its pairwise formula.

    `pairwise` maps the matrices of per-coordinate gaps |dx| and |dy| to the distance matrix.
    """

    key: str
    label: str
    pairwise: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


DISTANCES = {
    distance.key: distance
    for distance in (
        Distance("l2", "L2 distance", lambda dx, dy: np.sqrt(dx * dx + dy * dy)),
        Distance("x", "x distance", lambda dx, dy: dx),
        Distance("y", "y distance", lambda dx, dy: dy),
        Distance("linf", "L\u2212\u221e distance", np.minimum),
        Distance("geomean", "geometric-mean distance", lambda dx, dy: np.sqrt(dx * dy)),
    )
}
REFERENCE_KEYS = ("l2", "x", "y")  # the neighbors marked whatever the objective
REFERENCE_GLYPHS = {"l2": "dot", "x": "|", "y": "-"}


def nearest_neighbors(
    x: NDArray[np.floating], y: NDArray[np.floating], keys: tuple[str, ...]
) -> dict[str, tuple[NDArray[np.intp], NDArray[np.float64]]]:
    """Return, per distance key, each item's nearest other item and its distance to it.

    Two items sharing a coordinate are at distance 0 under the x, y, L-inf and geometric-mean distances;
    that pair is then each other's nearest neighbor, and the JavaScript draws the degenerate level curve.
    """
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    dx = np.abs(x64[:, None] - x64[None, :])
    dy = np.abs(y64[:, None] - y64[None, :])
    rows = np.arange(len(x64))
    result = {}
    for key in keys:
        matrix = DISTANCES[key].pairwise(dx, dy)
        np.fill_diagonal(matrix, np.inf)
        index = matrix.argmin(axis=1)
        result[key] = (index, matrix[rows, index])
    return result


# ==================================================================================================
#  SVG pieces
# ==================================================================================================
def px(x: float) -> float:
    """Return the pixel x of a data x."""
    return MARGIN_LEFT + (x - X_MIN) * SCALE


def py(y: float) -> float:
    """Return the pixel y of a data y (pixel y grows downward)."""
    return MARGIN_TOP + (Y_MAX - y) * SCALE


def population_raster_svg(population_image_url: str) -> list[str]:
    """Return the white background and the population raster over the unit square, in pixel space."""
    square_x, square_y = px(0.0), py(1.0)
    return [
        f'<rect x="0" y="0" width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}" fill="#ffffff"/>',
        f'<image href="{population_image_url}" x="{square_x:.2f}" y="{square_y:.2f}"'
        f' width="{SCALE:.2f}" height="{SCALE:.2f}" preserveAspectRatio="none"/>',
    ]


def data_group_open() -> str:
    """Return the opening `<g>` of the data layer, mapping data coordinates to pixels."""
    return f'<g class="usx-data" transform="translate({px(0.0):.2f},{py(0.0):.2f}) scale({SCALE:.3f},{-SCALE:.3f})">'


def axes_svg() -> list[str]:
    """Return the two spines with their ticks and tick labels, in pixel space."""
    left, bottom, top, right = px(X_MIN), py(Y_MIN), py(Y_MAX), px(X_MAX)
    parts = [
        f'<path class="usx-spine" d="M{left:.1f},{top:.1f}V{bottom:.1f}H{right:.1f}"/>',
    ]
    for tick in TICKS:
        x, y = px(tick), py(tick)
        parts.append(f'<line class="usx-tick" x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 4:.1f}"/>')
        parts.append(f'<text class="usx-label" x="{x:.1f}" y="{bottom + 17:.1f}" text-anchor="middle">{tick:g}</text>')
        parts.append(f'<line class="usx-tick" x1="{left:.1f}" y1="{y:.1f}" x2="{left - 4:.1f}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="usx-label" x="{left - 7:.1f}" y="{y:.1f}" text-anchor="end" dominant-baseline="middle">'
            f"{tick:g}</text>"
        )
    return parts


def _glyph(x: float, y: float, ring_radius: float, glyph: str) -> str:
    """Return the glyph inscribed in a ring of `ring_radius` at (x, y): a central dot or an axis-parallel stroke."""
    reach = GLYPH_REACH_FACTOR * ring_radius
    if glyph == "dot":
        return f'<circle class="usx-glyph-dot" cx="{x:.1f}" cy="{y:.1f}" r="{CENTER_DOT_FACTOR * ring_radius:.1f}"/>'
    elif glyph == "|":
        return f'<path class="usx-glyph" d="M{x:.1f},{y - reach:.1f}L{x:.1f},{y + reach:.1f}"/>'
    else:
        return f'<path class="usx-glyph" d="M{x - reach:.1f},{y:.1f}L{x + reach:.1f},{y:.1f}"/>'


def _legend_mark(x: float, y: float, glyph: str) -> str:
    """Return a legend mark: the dashed ring drawn around a reference neighbor, with its inscribed glyph."""
    ring = f'<circle class="usx-mark" cx="{x:.1f}" cy="{y:.1f}" r="{LEGEND_RING_RADIUS}"/>'
    return ring + _glyph(x, y, LEGEND_RING_RADIUS, glyph)


def legend_svg(n: int, k: int, objective_keys: tuple[str, ...], with_neighbor_marks: bool = True) -> list[str]:
    """Return the legend box in the top-right corner, in pixel space.

    The left column names what is always drawn; the right column names the reference neighbors the
    interaction marks around the picked item. The blue neighbor row appears only for a single-distance
    objective: a hybrid has one nearest neighbor per term, and those are the reference neighbors. A
    figure without the neighbor interaction (`with_neighbor_marks=False`) gets the left column's first
    two rows only.
    """
    column_widths, row, pad = (250, 190) if with_neighbor_marks else (170,), 18, 8
    width = sum(column_widths) + pad
    x0 = px(X_MAX) - 6 - width
    y0 = py(Y_MAX) + 6
    left_column = [
        (
            lambda x, y: f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="{POPULATION_COLOR}"/>',
            f"population (n = {n:,})",
        ),
        (
            lambda x, y: f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{SELECTION_COLOR}"/>',
            f"selection (k = {k})",
        ),
    ]
    if len(objective_keys) == 1 and with_neighbor_marks:
        left_column.append(
            (
                lambda x, y: f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{NEIGHBOR_COLOR}"/>',
                f"nearest neighbor, {DISTANCES[objective_keys[0]].label}",
            )
        )
    right_column = [
        (lambda x, y, g=REFERENCE_GLYPHS[key]: _legend_mark(x, y, g), f"nearest neighbor, {DISTANCES[key].label}")
        for key in REFERENCE_KEYS
    ]
    columns = (left_column, right_column) if with_neighbor_marks else (left_column,)
    height = pad * 2 + row * max(len(column) for column in columns)
    parts = [f'<rect class="usx-legend" x="{x0}" y="{y0}" width="{width}" height="{height}"/>']
    x = x0
    for column, column_width in zip(columns, column_widths):
        for i, (mark, label) in enumerate(column):
            y = y0 + pad + row * i + row / 2
            parts.append(mark(x + 14, y))
            parts.append(
                f'<text class="usx-label" x="{x + 26:.1f}" y="{y:.1f}" dominant-baseline="middle">{label}</text>'
            )
        x += column_width
    return parts


def _data_group(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    neighbors: dict[str, tuple[NDArray[np.intp], NDArray[np.float64]]],
) -> list[str]:
    """Return the group in data coordinates: the hover layer, the rug ticks and their hit areas, the dots, the marks.

    Each dot carries its neighbors as JSON in `data-nn`: distance key to `[neighbor index, distance]`.
    Children are listed in paint order: the marks layer comes last so rings draw over the dots.
    """
    parts = [
        data_group_open(),
        '<g class="usx-hover" clip-path="url(#usx-square)"></g>',
    ]
    for cls in ("usx-rug", "usx-hit"):
        for i, (xi, yi) in enumerate(zip(x, y)):
            parts.append(
                f'<line class="{cls} usx-rug-x" data-i="{i}" x1="{xi:.5f}" y1="{RUG_NEAR}" x2="{xi:.5f}" y2="{RUG_FAR}"'
                ' vector-effect="non-scaling-stroke"/>'
            )
            parts.append(
                f'<line class="{cls} usx-rug-y" data-i="{i}" x1="{RUG_NEAR}" y1="{yi:.5f}" x2="{RUG_FAR}" y2="{yi:.5f}"'
                ' vector-effect="non-scaling-stroke"/>'
            )
    for i, (xi, yi) in enumerate(zip(x, y)):
        record = {key: [int(index[i]), round(float(distance[i]), 5)] for key, (index, distance) in neighbors.items()}
        parts.append(
            f'<circle class="usx-dot" data-i="{i}" data-nn=\'{json.dumps(record, separators=(",", ":"))}\''
            f' cx="{xi:.5f}" cy="{yi:.5f}" r="{DOT_RADIUS}" vector-effect="non-scaling-stroke"'
            f' tabindex="0" role="button" aria-label="item {i}"/>'
        )
    parts.append('<g class="usx-marks"></g>')
    parts.append("</g>")
    return parts


def band_lines_svg(band_edges: tuple[float, ...]) -> list[str]:
    """Return a light line across the unit square at each band edge, along both axes, in pixel space."""
    left, right, bottom, top = px(0.0), px(1.0), py(0.0), py(1.0)
    parts = []
    for edge in band_edges:
        x, y = px(edge), py(edge)
        parts.append(f'<line class="usx-band" x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}"/>')
        parts.append(f'<line class="usx-band" x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}"/>')
    return parts


def explorer_fragment(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    n: int,
    k: int,
    objective_keys: tuple[str, ...],
    population_image_url: str,
    description: str,
    band_edges: tuple[float, ...] = (),
) -> str:
    """Return the HTML fragment: a `<div>` holding the SVG and its caption, ending in a newline.

    Args:
        x, y: Coordinates of the k selected items, in the solver's input coordinates.
        n: Population size, for the legend.
        k: Selection size, for the legend.
        objective_keys: Keys into `DISTANCES` of the distances the experiment's objective uses; one for a
            simple objective, one per term for a hybrid. The interaction draws one level curve per key.
        population_image_url: URL of the population raster, relative to the page that includes the fragment.
        description: Alternative text of the figure.
        band_edges: Interior edges of the bands that a constrained experiment cuts each axis into, drawn as light
            lines across the square along both axes; empty for an unconstrained experiment.
    """
    keys = objective_keys + tuple(key for key in REFERENCE_KEYS if key not in objective_keys)
    neighbors = nearest_neighbors(x, y, keys)
    labels = {key: DISTANCES[key].label for key in keys}
    lines = [
        '<div class="usx-figure">',
        f'<svg class="usx" viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" xmlns="http://www.w3.org/2000/svg" role="img"'
        f" data-objective=\"{' '.join(objective_keys)}\" data-labels='{json.dumps(labels, separators=(',', ':'))}'"
        f' data-ring="{RING_RADIUS_FACTOR}" data-reach="{GLYPH_REACH_FACTOR}" data-center="{CENTER_DOT_FACTOR}">',
        f"<title>Selection maximizing diversity under the {', '.join(labels[key] for key in objective_keys)}</title>",
        f"<desc>{description}</desc>",
        '<defs><clipPath id="usx-square" clipPathUnits="userSpaceOnUse">'
        '<rect x="0" y="0" width="1" height="1"/></clipPath></defs>',
        *population_raster_svg(population_image_url),
        *band_lines_svg(band_edges),
        *axes_svg(),
        *legend_svg(n, k, objective_keys),
        *_data_group(x, y, neighbors),
        "</svg>",
        f'<p class="usx-caption">{HINT}</p>',
        "</div>",
    ]
    return "\n".join(lines) + "\n"
