"""Build the solve replay figure of `docs/guides/uniform_sampling.md` as an HTML fragment.

The fragment is an inline SVG over a raster of the population, with the axes, band lines and legend of
the experiment figures (`uniform_sampling_explorer.py`), followed by a slider, two step buttons and a
caption. The SVG carries the coordinates of every item selected in any frame in `data-points`, and
every frame as a list of indices into that table in `data-frames`; only the last frame, the final
selection, is drawn into the fragment, so that it shows without JavaScript, and
`docs/javascripts/uniform_sampling_replay.js` redraws the dots and rug ticks for any other frame.

This module depends on numpy only, so its tests run without the `benchmarks` dependency group.
"""

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from uniform_sampling_explorer import (
    DISTANCES,
    DOT_RADIUS,
    RUG_FAR,
    RUG_NEAR,
    SCALE,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    axes_svg,
    band_lines_svg,
    legend_svg,
    px,
    py,
)

HINT = "Each frame is a checkpoint at which the selection changed. Drag the slider, or use the buttons or the arrow keys."


@dataclass(frozen=True)
class ReplayFrame:
    """A frame is the best selection held at one checkpoint: when it was held, how diverse it was, which items."""

    t_sec: float
    diversity: float
    items: list[int]


def caption(index: int, n_frames: int, frame: ReplayFrame) -> str:
    """Return a frame's caption line; the JavaScript formats every other frame's the same way."""
    return f"frame {index + 1}/{n_frames} · {frame.t_sec:.1f} s · diversity {frame.diversity:.4f}"


def item_svg(position: int, x: float, y: float) -> list[str]:
    """Return one drawn item, in data coordinates: its two rug ticks and its dot, grouped under its table position.

    The JavaScript draws the same three elements for the items of any other frame.
    """
    return [
        f'<g class="usx-item" data-p="{position}">',
        f'<line class="usx-rug usx-rug-x" x1="{x:.4f}" y1="{RUG_NEAR}" x2="{x:.4f}" y2="{RUG_FAR}"'
        ' vector-effect="non-scaling-stroke"/>',
        f'<line class="usx-rug usx-rug-y" x1="{RUG_NEAR}" y1="{y:.4f}" x2="{RUG_FAR}" y2="{y:.4f}"'
        ' vector-effect="non-scaling-stroke"/>',
        f'<circle class="usx-dot" cx="{x:.4f}" cy="{y:.4f}" r="{DOT_RADIUS}"/>',
        "</g>",
    ]


def _controls(n_frames: int) -> list[str]:
    """Return the slider and the two step buttons, the slider at the last frame."""
    return [
        '<div class="usx-controls">',
        '<button type="button" class="usx-step" data-step="-1" aria-label="previous frame">&#9664;</button>',
        f'<input type="range" class="usx-slider" min="0" max="{n_frames - 1}" value="{n_frames - 1}"'
        ' aria-label="frame"/>',
        '<button type="button" class="usx-step" data-step="1" aria-label="next frame">&#9654;</button>',
        "</div>",
    ]


def replay_fragment(
    vectors: NDArray[np.floating],
    frames: list[ReplayFrame],
    n: int,
    k: int,
    objective_keys: tuple[str, ...],
    population_image_url: str,
    description: str,
    band_edges: tuple[float, ...] = (),
) -> str:
    """Return the HTML fragment: a `<div>` holding the SVG, the controls and the caption, ending in a newline.

    Args:
        vectors: The population's coordinates, in the solver's input coordinates; frames index into it.
        frames: The selections to step through, in solve order; at least one.
        n: Population size, for the legend.
        k: Selection size, for the legend.
        objective_keys: Keys into `DISTANCES` of the distances the objective uses, for the title.
        population_image_url: URL of the population raster, relative to the page that includes the fragment.
        description: Alternative text of the figure.
        band_edges: Interior edges of the bands that a constrained experiment cuts each axis into, drawn as light
            lines across the square along both axes; empty for an unconstrained experiment.

    Raises:
        ValueError: If `frames` is empty.
    """
    if not frames:
        raise ValueError("a replay needs at least one frame")
    items = sorted({item for frame in frames for item in frame.items})
    position = {item: index for index, item in enumerate(items)}
    points = [[round(float(vectors[item, 0]), 4), round(float(vectors[item, 1]), 4)] for item in items]
    frame_records = [[frame.t_sec, round(frame.diversity, 6), [position[item] for item in frame.items]] for frame in frames]
    last = frames[-1]
    labels = [DISTANCES[key].label for key in objective_keys]
    square_x, square_y = px(0.0), py(1.0)
    lines = [
        '<div class="usx-figure usx-replay" tabindex="0">',
        f'<svg class="usx" viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" xmlns="http://www.w3.org/2000/svg" role="img"'
        f" data-points='{json.dumps(points, separators=(',', ':'))}'"
        f" data-frames='{json.dumps(frame_records, separators=(',', ':'))}'"
        f' data-rug="{RUG_NEAR} {RUG_FAR}" data-r="{DOT_RADIUS}">',
        f"<title>Selections held during a solve maximizing diversity under the {', '.join(labels)}</title>",
        f"<desc>{description}</desc>",
        f'<rect x="0" y="0" width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}" fill="#ffffff"/>',
        f'<image href="{population_image_url}" x="{square_x:.2f}" y="{square_y:.2f}"'
        f' width="{SCALE:.2f}" height="{SCALE:.2f}" preserveAspectRatio="none"/>',
        *band_lines_svg(band_edges),
        *axes_svg(),
        *legend_svg(n, k, objective_keys, with_neighbor_marks=False),
        f'<g class="usx-data" transform="translate({px(0.0):.2f},{py(0.0):.2f}) scale({SCALE:.3f},{-SCALE:.3f})">',
        *(line for item in last.items for line in item_svg(position[item], *points[position[item]])),
        "</g>",
        "</svg>",
        *_controls(len(frames)),
        f'<p class="usx-caption usx-frame">{caption(len(frames) - 1, len(frames), last)}</p>',
        f'<p class="usx-caption">{HINT}</p>',
        "</div>",
    ]
    return "\n".join(lines) + "\n"
