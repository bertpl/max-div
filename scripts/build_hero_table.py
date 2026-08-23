"""Render the README hero capability table to light and dark SVGs.

Reads the same capability data as the documentation surfaces — the axes file, the solver registry
and the per-tool capability records — and writes the light and dark hero SVGs into ``OUT_DIR``.
Every cell the README shows therefore traces back to the record that defends it, and a capability
cannot say one thing here and another on the comparison page.

Validation belongs to scripts/capability_data.py: this renders what it is given, and data that
would not pass there is reported by the tool that knows the rules rather than half-caught here.

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

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
OUT_DIR = REPO_ROOT / "docs" / "images"

# `scripts/` is maintainer tooling rather than an importable package, so the sibling loader is
# reached by putting this directory on the path. Running the script does that already; importing
# it, as the tests do, does not.
sys.path.insert(0, str(SCRIPTS_DIR))

import capability_data  # noqa: E402

# --- geometry (all integers; no float formatting anywhere) --
LABEL_W = 148  # solver-name gutter
COL_W = 26  # one mark column
SCALE_W = 46  # one measured-scaling column — wide enough for the longest suffix value ("500M")
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

# How each drawn mark is painted: its palette key, and a font weight where it needs one. The glyphs
# and the legend wording come from the axes file; only the styling is this surface's own business,
# so it sits with the themes. A mark the hero draws as nothing needs no entry.
MARK_STYLE = {"full": ("mark", "700"), "partial": ("partial", None)}


# ==================================================================================================
#  HeroTable
# ==================================================================================================
class HeroTable:
    """The hero's view of the capability data: its columns, its rows, and its mark vocabulary.

    The README table is a narrower reading of what the documentation surfaces render — only the
    hero-visible axes, only the short labels, and none of the metadata. Resolving that view once,
    here, lets the geometry below speak in columns and rows rather than in axes and records.
    """

    def __init__(self, axes: dict, registry: dict, records: dict):
        self.marks = axes["marks"]
        axis_keys = [
            f"{group['key']}.{axis['key']}" for group in axes["groups"] for axis in group["axes"] if axis.get("hero")
        ]
        self.groups = [
            (group["hero_label"], [(axis["hero_label"], 1) for axis in group["axes"] if axis.get("hero")])
            for group in axes["groups"]
        ]
        # The scaling columns are not mark columns: each is wider, and its cell is a measured size
        # (or the pending marker) rather than a glyph. They form one group so the band that carries
        # the scale-columns band label is drawn by the same code as every other group's.
        scale_columns = axes["scale_columns"]["columns"]
        self.groups.append((axes["scale_columns"]["hero_label"], [(c["hero_label"], 2) for c in scale_columns]))
        self.categories = [category["label"] for category in registry["categories"]]
        self.rows = [
            {
                "category": index,
                "name": tool.get("hero_name") or tool["name"],
                "subject": bool(tool.get("subject")),
                "marks": [records[tool["key"]][0]["capabilities"][key]["mark"] for key in axis_keys],
                "scales": [records[tool["key"]][0]["scale"][c["key"]] for c in scale_columns],
            }
            for index, category in enumerate(registry["categories"])
            for tool in category["tools"]
        ]

    @classmethod
    def from_repo(cls) -> "HeroTable":
        """Build the table from the committed data files."""
        return cls(capability_data.load_axes(), capability_data.load_registry(), capability_data.load_records())


def esc(text):
    """Escape the five XML entities."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


SCALE_FS = 12  # base size for the scaling figures
NAME_FS = "12.5"  # solver-name size
MARK_FS = "13.5"  # check-mark and tilde size
SMALL_FS = 11  # group labels, category labels, caption
PENDING_GLYPH = "\u2026"  # a scaling cell awaiting its measurement
DAGGER = "\u2020"  # anchors the scaling column headers to the footnote row under the table
SCALING_FOOTNOTE = "based on the built-in benchmark problem U1 and the published measurement protocol"
CAPTION2_H = 16  # the footnote row under the mark-legend row


def scale_text(value):
    """Return one scaling cell's text: the measured size in suffix notation, or the pending marker.

    The notation is `format_scale_value`'s; sharing the formatter guarantees the hero and the
    documentation tables cannot print one value two ways.
    """
    return PENDING_GLYPH if value == capability_data.PENDING else capability_data.format_scale_value(int(value))


class _Layout:
    """Resolved geometry and palette for one render. Every value is an integer."""

    def __init__(self, table, theme_name):
        self.t = THEMES[theme_name]
        self.marks = table.marks
        self.groups = table.groups
        self.rows = table.rows
        self.categories = table.categories
        flat = [(h, w) for _, cols in self.groups for (h, w) in cols]
        self.mark_cols = [(h, w) for h, w in flat if w == 1]
        self.scale_cols = [(h, w) for h, w in flat if w != 1]
        self.n_marks = len(self.mark_cols)
        self.grid_w = self.n_marks * COL_W + len(self.scale_cols) * SCALE_W

        # The longest header, rotated 45 degrees, sticks out to the right by width/sqrt(2). That
        # same overhang sizes the header band and the skew of the group bands.
        longest = max(len(h) for h, _ in self.mark_cols + self.scale_cols)
        self.overhang = (longest * CHAR_W * 7) // 100  # /10 for tenths, *0.707 for the rotation
        self.header_h = self.overhang + 18

        self.width = PAD + LABEL_W + self.grid_w + self.overhang + PAD
        self.height = (
            PAD
            + GROUP_H
            + self.header_h
            + len(self.categories) * CAT_H
            + len(self.rows) * ROW_H
            + CAPTION_H
            + CAPTION2_H
            + BOTTOM_PAD
        )

        self.y_group = PAD + GROUP_H - 6
        self.y_top = PAD + GROUP_H  # bands start here, just under the group labels
        self.y_header_base = PAD + GROUP_H + self.header_h
        self.y_end = self.y_header_base + len(self.categories) * CAT_H + len(self.rows) * ROW_H
        # The bands turn from vertical to 45 degrees a little above the first row, so the corner
        # sits clear of the top row rather than flush against it.
        self.y_corner = self.y_header_base - CORNER_LIFT
        self.skew = self.y_corner - self.y_top  # a 45-degree edge shifts right by its own height
        self.table_w = LABEL_W + self.grid_w

    def leads_on_scale(self, row, j):
        """Return True when this row carries scaling column j's highest measured value.

        Deliberately not keyed to max-div: whoever measures highest leads, and saying so is the
        point of showing the columns at all. A pending cell never leads, and a column with no
        measured value yet has no leader.
        """
        measured = [int(r["scales"][j]) for r in self.rows if r["scales"][j] != capability_data.PENDING]
        return bool(measured) and row["scales"][j] != capability_data.PENDING and int(row["scales"][j]) == max(measured)

    def any_pending(self):
        """Return True while any scaling cell awaits its measurement.

        The pending legend entry is drawn only while one does.
        """
        return any(value == capability_data.PENDING for row in self.rows for value in row["scales"])

    def col_x(self, i):
        """Return the left edge of mark column i."""
        return PAD + LABEL_W + i * COL_W

    def scale_x(self, j):
        """Return the left edge of scaling column j, which sits right of every mark column."""
        return PAD + LABEL_W + self.n_marks * COL_W + j * SCALE_W

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
    edges = set()
    for gi, (glabel, cols) in enumerate(lay.groups):
        span = sum(COL_W if w == 1 else SCALE_W for _, w in cols)
        x0 = lay.col_x(idx)
        x1 = x0 + span
        edges.update((x0, x1))
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
    # A hairline where one group ends and the next begins, and on the outer edges of the first
    # and last group — the bracket rules' color at half their weight, so the verticals read as
    # quieter scaffolding. Each follows its band's edge: slanted alongside the rotated labels,
    # then vertical from the corner down to the last row.
    for x in sorted(edges):
        pts = f"{x + lay.skew},{lay.y_top} {x},{lay.y_corner} {x},{lay.y_end}"
        out.append(f'<polyline points="{pts}" fill="none" stroke="{t["rule"]}" stroke-width="0.5"/>')
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
    for j, (header, _w) in enumerate(lay.scale_cols):
        x = lay.scale_x(j) + SCALE_W // 2 - 3
        out.append(
            f'<text x="{x}" y="{y}" fill="{lay.t["ink"]}" font-size="{HEADER_FS}" '
            f'text-anchor="start" transform="rotate(-45 {x} {y})">{esc(header + " " + DAGGER)}</text>'
        )
    return out


def _row_marks(lay, row, y):
    """The mark glyphs for one row. A mark the hero draws as nothing leaves its cell empty."""
    t, out = lay.t, []
    cy = y + ROW_H - 7
    for i, mark in enumerate(row["marks"]):
        glyph = lay.marks[mark]["hero_glyph"]
        if not glyph:
            continue
        color, weight = MARK_STYLE[mark]
        bold = f' font-weight="{weight}"' if weight else ""
        cx = lay.col_x(i) + COL_W // 2
        out.append(
            f'<text x="{cx}" y="{cy}" fill="{t[color]}" font-size="{MARK_FS}"{bold} '
            f'text-anchor="middle">{esc(glyph)}</text>'
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
            own = row["subject"]
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
            for j, value in enumerate(row["scales"]):
                pending = value == capability_data.PENDING
                leads = lay.leads_on_scale(row, j)
                fill = t["partial"] if pending else (t["mark"] if leads else t["ink"])
                # Bold in a scaling column means "leads this column" and nothing else. The
                # row-name weight is a separate signal (the subject), so inheriting it here would
                # imply max-div leads a column it does not.
                weight = "700" if leads else "400"
                out.append(
                    f'<text x="{lay.scale_x(j) + SCALE_W // 2}" y="{y + ROW_H - 7}" fill="{fill}" '
                    f'font-size="{SCALE_FS}" font-weight="{weight}" text-anchor="middle">'
                    f"{esc(scale_text(value))}</text>"
                )
            y += ROW_H
    return out, y


def _legend(lay, y):
    """The two legend rows under the table: the marks, then the scaling footnote and pending marker.

    Both glyphs are bolded here whatever weight they carry in the grid: at legend size they sit
    inline in a line of prose, where the grid's lighter tilde would disappear.
    """
    t, spans = lay.t, []
    for mark, spec in lay.marks.items():
        if not spec["hero_glyph"]:
            continue
        gap = ' dx="9"' if spans else ""
        if spans:
            spans.append(f'<tspan fill="{t["muted"]}" dx="9">·</tspan>')
        spans.append(f'<tspan fill="{t[MARK_STYLE[mark][0]]}" font-weight="700"{gap}>{esc(spec["hero_glyph"])}</tspan>')
        spans.append(f'<tspan fill="{t["muted"]}" dx="5">{esc(spec["legend"])}</tspan>')
    notes = [
        f'<tspan fill="{t["muted"]}" font-weight="700">{esc(DAGGER)}</tspan>',
        f'<tspan fill="{t["muted"]}" dx="5">{esc(SCALING_FOOTNOTE)}</tspan>',
    ]
    if lay.any_pending():
        notes.append(f'<tspan fill="{t["muted"]}" dx="9">·</tspan>')
        notes.append(f'<tspan fill="{t["partial"]}" font-weight="700" dx="9">{esc(PENDING_GLYPH)}</tspan>')
        notes.append(f'<tspan fill="{t["muted"]}" dx="5">measurement pending</tspan>')
    return [
        f'<text x="{PAD}" y="{y + CAPTION_H - 8}" font-size="{SMALL_FS}">' + "".join(spans) + "</text>",
        f'<text x="{PAD}" y="{y + CAPTION_H + CAPTION2_H - 8}" font-size="{SMALL_FS}">' + "".join(notes) + "</text>",
    ]


def build_svg(table, theme_name):
    """Return the SVG document for one theme."""
    lay = _Layout(table, theme_name)
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

    table = HeroTable.from_repo()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for theme in THEMES:
        target = OUT_DIR / f"hero_{theme}.svg"
        svg = build_svg(table, theme)
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
