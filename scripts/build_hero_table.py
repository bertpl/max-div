"""Render the README hero capability table to light and dark SVGs.

Reads scripts/hero_table_data.txt and writes docs/images/hero_light.svg and
docs/images/hero_dark.svg.

Two properties are deliberate:

* **Each SVG carries its own opaque background.** GitHub exposes no hook for its own theme
  setting, so `<picture>` + `prefers-color-scheme` follows the reader's operating system. A
  reader who pinned GitHub's theme against their OS gets the other variant, and an opaque
  panel keeps it legible instead of putting dark ink on a dark page.
* **Output is deterministic.** Coordinates are rounded before formatting and floats are never
  repr'd, so re-running reproduces the committed bytes and a drift test can compare them.

Usage: python scripts/build_hero_table.py [--check]
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "scripts" / "hero_table_data.txt"
OUT_DIR = REPO_ROOT / "docs" / "images"

# --- column definitions ------------------------------------
# (group label, [(short header, width)]) — headers render at 45 degrees.
GROUPS = [
    ("distance", [("L1", 1), ("L2", 1), ("cosine", 1), ("custom", 1)]),
    ("objective", [("max-min", 1), ("mean-of-NN", 1), ("geomean-of-NN", 1), ("max-sum", 1)]),
    ("constraints", [("disjoint groups", 1), ("overlapping groups", 1), ("ranged counts", 1)]),
    ("budget", [("iterations", 1), ("wall clock", 1), ("improves with budget", 1)]),
    ("max practical n", [("", 2)]),
]

# --- geometry (all integers; no float formatting anywhere) --
LABEL_W = 148  # solver-name gutter
COL_W = 26  # one mark column
SCALE_W = 90  # wide enough for the group label "max practical n" to sit over it without colliding
ROW_H = 23
HEADER_H = 132  # room for the 45-degree labels
GROUP_H = 18
CAPTION_H = 21  # band under the table holding the mark legend
CAT_H = 20
BOTTOM_PAD = 5  # less than PAD: the legend should sit close under the table, not centred
CORNER_LIFT = 15  # where the band edge turns diagonal, above the first row
PAD = 14
HEADER_FS = 12  # header font size; the width estimate below is calibrated to it
CHAR_W = 63  # tenths of a px per character at HEADER_FS, averaged over mixed-case text

THEMES = {
    "light": {
        "bg": "#ffffff",
        "panel": "#ffffff",
        "ink": "#1f2328",
        "muted": "#656d76",
        "rule": "#d8dee4",
        "band": "#f6f8fa",
        "mark": "#1a7f37",
        "partial": "#8c959f",
        "own_band": "#ddf4e4",
        "cat_band": "#57606a",
        "cat_band_opacity": "0.06",
    },
    "dark": {
        "bg": "#0d1117",
        "panel": "#0d1117",
        "ink": "#e6edf3",
        "muted": "#8b949e",
        "rule": "#30363d",
        "band": "#161b22",
        "mark": "#3fb950",
        "partial": "#8b949e",
        "own_band": "#12261e",
        "cat_band": "#c9d1d9",
        "cat_band_opacity": "0.05",
    },
}


def parse_data(text):
    """Parse the companion file into (categories, rows). Rows carry marks in column order."""
    categories, rows = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[category]"):
            categories.append(line[len("[category]") :].strip())
            continue
        name, *rest = [p.strip() for p in line.split("|")]
        *mark_groups, scale, source = rest
        marks = []
        for grp in mark_groups:
            marks.extend(grp.split())
        rows.append(
            {
                "category": len(categories) - 1,
                "name": name,
                "marks": marks,
                "scale": scale,
                "source": source,
            }
        )
    n_cols = sum(len(cols) for _, cols in GROUPS) - 1  # scale is not a mark column
    for row in rows:
        if len(row["marks"]) != n_cols:
            raise ValueError(f"{row['name']}: expected {n_cols} marks, got {len(row['marks'])}")
    return categories, rows


def esc(text):
    """Escape the five XML entities."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


SCALE_FS = 12  # base size for the scale figures
NAME_FS = "12.5"  # solver-name size
MARK_FS = "13.5"  # check-mark and tilde size
SMALL_FS = 11  # group labels, category labels, caption
SUP_FS = 8  # exponent size — unicode superscripts render around 7px here, which reads too small
SUP_DY = 4  # how far the exponent is raised


def scale_markup(spec):
    """`4-5` -> `~10^4 en-dash 10^5` as SVG markup, `3` -> `~10^3`.

    The leading tilde is the same glyph as the `partial` mark, which is a deliberate choice
    rather than an oversight: an approximation sign was tried and read worse. The two senses are
    told apart by column, and by the marks being grey while the figures are ink or green.

    Exponents are `<tspan>`s rather than unicode superscript characters: those are locked to
    roughly 0.6x the surrounding size, which is too small to read at this scale. `dy` shifts are
    cumulative in SVG, so every raised span is followed by an equal lowering span.

    The figures are indicative orders of magnitude, not measured ceilings, and the binding limit
    differs per tool (memory, runtime, dimensionality).
    """

    def sup(digits):
        return f'<tspan font-size="{SUP_FS}" dy="-{SUP_DY}">{digits}</tspan>'

    def baseline(text):
        return f'<tspan font-size="{SCALE_FS}" dy="{SUP_DY}">{text}</tspan>'

    if "-" in spec:
        lo, hi = spec.split("-")
        return "~10" + sup(lo) + baseline("\u201310") + sup(hi)
    return "~10" + sup(spec)


class _Layout:
    """Resolved geometry and palette for one render. Every value is an integer."""

    def __init__(self, categories, rows, theme_name):
        self.t = THEMES[theme_name]
        self.rows = rows
        self.categories = categories
        self.mark_cols = [(h, w) for _, cols in GROUPS for (h, w) in cols][:-1]
        self.n_marks = len(self.mark_cols)
        self.grid_w = self.n_marks * COL_W + SCALE_W

        # The longest header, rotated 45 degrees, sticks out to the right by width/sqrt(2). That
        # same overhang sizes the header band and the skew of the group bands.
        longest = max(len(h) for h, _ in self.mark_cols)
        self.overhang = (longest * CHAR_W * 7) // 100  # /10 for tenths, *0.707 for the rotation
        self.header_h = self.overhang + 18

        self.width = PAD + LABEL_W + self.grid_w + self.overhang + PAD
        self.height = (
            PAD + GROUP_H + self.header_h + len(categories) * CAT_H + len(rows) * ROW_H + CAPTION_H + BOTTOM_PAD
        )

        self.y_group = PAD + GROUP_H - 6
        self.y_top = PAD + GROUP_H  # bands start here, just under the group labels
        self.y_header_base = PAD + GROUP_H + self.header_h
        self.y_end = self.y_header_base + len(categories) * CAT_H + len(rows) * ROW_H
        # The bands turn from vertical to 45 degrees a little above the first row, so the corner
        # sits clear of the top row rather than flush against it.
        self.y_corner = self.y_header_base - CORNER_LIFT
        self.skew = self.y_corner - self.y_top  # a 45-degree edge shifts right by its own height
        self.table_w = LABEL_W + self.grid_w

    def leads_on_scale(self, row):
        """True for the row(s) carrying the highest scale figure.

        Deliberately not keyed to max-div: the leader here is a competitor, and saying so is the
        point of showing the column at all.
        """
        best = max(int(r["scale"].split("-")[-1]) for r in self.rows)
        return int(row["scale"].split("-")[-1]) == best

    def col_x(self, i):
        """Left edge of mark column i."""
        return PAD + LABEL_W + i * COL_W

    def rule(self, y):
        """A full-width horizontal rule at y."""
        return (
            f'<line x1="{PAD}" y1="{y}" x2="{PAD + self.table_w}" y2="{y}" stroke="{self.t["rule"]}" stroke-width="1"/>'
        )


def _group_bands(lay):
    """Alternating group bands, their labels, and the bracket rule under each label.

    Each band runs from the group label down to the last row. Through the header region its top
    edge is slanted 45 degrees, parallel to the rotated column labels, so every label stays
    inside its own column's band the whole way up. The consequence is that the top of a band
    sits `skew` px right of its bottom, which is why the labels are offset by the same amount.
    """
    out, idx, t = [], 0, lay.t
    for gi, (glabel, cols) in enumerate(GROUPS):
        span = sum(COL_W if w == 1 else SCALE_W for _, w in cols)
        x0 = lay.col_x(idx)
        x1 = x0 + span
        if gi % 2 == 0:
            corners = [
                (x0 + lay.skew, lay.y_top),
                (x1 + lay.skew, lay.y_top),
                (x1, lay.y_corner),
                (x1, lay.y_end),
                (x0, lay.y_end),
                (x0, lay.y_corner),
            ]
            pts = " ".join(f"{x},{y}" for x, y in corners)
            out.append(f'<polygon points="{pts}" fill="{t["band"]}"/>')
        lx = x0 + span // 2 + lay.skew
        out.append(
            f'<text x="{lx}" y="{lay.y_group}" fill="{t["muted"]}" font-size="{SMALL_FS}" '
            f'font-weight="600" text-anchor="middle">{esc(glabel)}</text>'
        )
        out.append(
            f'<line x1="{x0 + lay.skew + 2}" y1="{lay.y_group + 5}" x2="{x1 + lay.skew - 2}" '
            f'y2="{lay.y_group + 5}" stroke="{t["rule"]}" stroke-width="1"/>'
        )
        idx += len(cols)
    return out


def _column_headers(lay):
    """The rotated column headers.

    anchor=start + rotate(-45) leans each label up and to the right, the usual convention for
    narrow columns. SVG's y axis points down, so rotate(-45) is the counter-clockwise direction:
    anchor=end would send the text down into the data rows instead.
    """
    y = lay.y_header_base - 7
    out = []
    for i, (header, _w) in enumerate(lay.mark_cols):
        x = lay.col_x(i) + COL_W // 2 - 3
        out.append(
            f'<text x="{x}" y="{y}" fill="{lay.t["ink"]}" font-size="{HEADER_FS}" '
            f'text-anchor="start" transform="rotate(-45 {x} {y})">{esc(header)}</text>'
        )
    # the scale column gets no diagonal header — its group label already reads "max practical n"
    return out


def _row_marks(lay, row, y):
    """The mark glyphs and the scale figure for one row."""
    t, out = lay.t, []
    cy = y + ROW_H - 7
    for i, mark in enumerate(row["marks"]):
        cx = lay.col_x(i) + COL_W // 2
        if mark == "Y":
            out.append(
                f'<text x="{cx}" y="{cy}" fill="{t["mark"]}" font-size="{MARK_FS}" '
                f'font-weight="700" text-anchor="middle">✓</text>'
            )
        elif mark == "~":
            out.append(
                f'<text x="{cx}" y="{cy}" fill="{t["partial"]}" font-size="{MARK_FS}" text-anchor="middle">~</text>'
            )
    return out


def _data_rows(lay):
    """Category bands and the solver rows beneath each, in declaration order."""
    t, out = lay.t, []
    y = lay.y_header_base
    for ci, category in enumerate(lay.categories):
        # A semi-transparent band behind every category label, so all categories read as peers —
        # otherwise a one-row category looks like a caption for the row beneath it. Translucent on
        # purpose: the vertical group bands stay visible through it.
        out.append(
            f'<rect x="{PAD}" y="{y}" width="{lay.table_w}" height="{CAT_H}" '
            f'fill="{t["cat_band"]}" fill-opacity="{t["cat_band_opacity"]}"/>'
        )
        out.append(
            f'<text x="{PAD}" y="{y + CAT_H - 6}" fill="{t["muted"]}" font-size="{SMALL_FS}" '
            f'font-weight="600" letter-spacing="0.5">{esc(category.upper())}</text>'
        )
        y += CAT_H
        for row in [r for r in lay.rows if r["category"] == ci]:
            own = row["name"] == "max-div"
            # `own` drives exactly two things: this weight, applied to the row name, and the band
            # below. The mark glyphs are styled uniformly across all rows.
            weight = "700" if own else "400"
            if own:
                out.append(f'<rect x="{PAD}" y="{y}" width="{lay.table_w}" height="{ROW_H}" fill="{t["own_band"]}"/>')
            out.append(lay.rule(y))
            out.append(
                f'<text x="{PAD + 2}" y="{y + ROW_H - 7}" fill="{t["ink"]}" '
                f'font-size="{NAME_FS}" font-weight="{weight}">{esc(row["name"])}</text>'
            )
            out.extend(_row_marks(lay, row, y))
            leads = lay.leads_on_scale(row)
            scale_fill = t["mark"] if leads else t["ink"]
            # Bold in this column means "leads on scale" and nothing else. The row-name weight is
            # a separate signal (the subject), so inheriting it here would imply max-div leads a
            # column it does not.
            scale_weight = "700" if leads else "400"
            out.append(
                f'<text x="{lay.col_x(lay.n_marks) + SCALE_W // 2}" y="{y + ROW_H - 7}" fill="{scale_fill}" '
                f'font-size="{SCALE_FS}" font-weight="{scale_weight}" text-anchor="middle">'
                f"{scale_markup(row['scale'])}</text>"
            )
            y += ROW_H
    return out, y


def _legend(lay, y):
    """The mark legend under the table."""
    t = lay.t
    return [
        f'<text x="{PAD}" y="{y + CAPTION_H - 8}" font-size="{SMALL_FS}">'
        f'<tspan fill="{t["mark"]}" font-weight="700">✓</tspan>'
        f'<tspan fill="{t["muted"]}" dx="5">built in</tspan>'
        f'<tspan fill="{t["muted"]}" dx="9">·</tspan>'
        f'<tspan fill="{t["partial"]}" font-weight="700" dx="9">~</tspan>'
        f'<tspan fill="{t["muted"]}" dx="5">reachable, but you supply the model, transform or metric</tspan>'
        f"</text>"
    ]


def build_svg(categories, rows, theme_name):
    """Return the SVG document for one theme."""
    lay = _Layout(categories, rows, theme_name)
    font = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{lay.width}" height="{lay.height}" '
        f'viewBox="0 0 {lay.width} {lay.height}" font-family="{font}">',
        f'<rect width="{lay.width}" height="{lay.height}" fill="{lay.t["bg"]}"/>',
    ]
    out += _group_bands(lay)
    out += _column_headers(lay)
    rows_out, y = _data_rows(lay)
    out += rows_out
    out.append(lay.rule(y))
    out += _legend(lay, y)
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed SVGs differ from a fresh render")
    args = parser.parse_args()

    categories, rows = parse_data(DATA_FILE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for theme in THEMES:
        target = OUT_DIR / f"hero_{theme}.svg"
        svg = build_svg(categories, rows, theme)
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != svg:
                stale.append(target.name)
        else:
            target.write_text(svg, encoding="utf-8")
            print(f"wrote {target.relative_to(REPO_ROOT)}")
    if args.check:
        if stale:
            print(f"stale (re-run scripts/build_hero_table.py): {', '.join(stale)}", file=sys.stderr)
            return 1
        print("hero SVGs are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
