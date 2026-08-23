"""Load, validate and render the solver capability data.

Three kinds of file feed this, split by what they describe:

* ``data/capability_axes.yaml`` — the columns: which axes exist, their labels, hero visibility.
* ``data/solver_registry.yaml`` — the rows: categories in order, tools in order within each.
* the per-tool records under ``RECORDS_DIR`` (one ``<key>.md`` each) — the cells and the prose.
  Each record's front matter is its data; its body is its profile page. One file, so a
  capability and the text defending it cannot drift apart.

Generated feature tables are written under ``generated/features/`` and pulled into each page by
a snippet line, which keeps the boundary between hand-authored and generated content visible. The
comparison page's two tables come from the same records and are written the same way, to
``generated/comparison.md`` — so the capability grid read across tools and the feature table read
down one tool cannot disagree. The capability-definitions page's body is generated the same way,
to ``generated/capability_definitions.md``, from the ``definition`` fields in the axes file — so
the criterion a column is judged by and the tables that judge by it share one source. The README
hero SVGs are rendered from these same three files by ``scripts/build_hero_table.py``, which is
why the rules below are worth enforcing here rather than per surface.

Two severities are reported. **Structural problems always fail**: an unknown axis, a missing
cell, a mark outside the vocabulary, a record that is not registered (or a registration with no
record), a scaling cell that is neither ``pending`` nor on the 1-2-5 grid, a column with no
definition, a page that forgot its include. **Near-duplicate note text also fails**, because two
notes that differ only in wording produce two footnotes where one was meant; a note may opt out
with ``distinct: true`` when the resemblance really is coincidental.

Usage:
    python scripts/capability_data.py            # validate, then write the fragments
    python scripts/capability_data.py --check    # validate and report only, write nothing
"""

import argparse
import re
import string
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
AXES_FILE = REPO_ROOT / "data" / "capability_axes.yaml"
REGISTRY_FILE = REPO_ROOT / "data" / "solver_registry.yaml"
RECORDS_DIR = REPO_ROOT / "docs" / "benchmarks" / "third_party" / "solvers"
FRAGMENTS_DIR = REPO_ROOT / "generated" / "features"
COMPARISON_FRAGMENT = REPO_ROOT / "generated" / "comparison.md"
COMPARISON_PAGE = REPO_ROOT / "docs" / "benchmarks" / "third_party" / "comparison.md"
COMPARISON_INCLUDE = '--8<-- "generated/comparison.md"'
DEFINITIONS_FRAGMENT = REPO_ROOT / "generated" / "capability_definitions.md"
DEFINITIONS_PAGE = REPO_ROOT / "docs" / "benchmarks" / "third_party" / "capability_definitions.md"
DEFINITIONS_INCLUDE = '--8<-- "generated/capability_definitions.md"'

# A measured scaling value sits on the protocol's 1-2-5 grid in n, smallest size 20 — the first
# alternative admits the two sub-100 grid sizes without admitting 10.
GRID_VALUE = re.compile(r"^(?:[25]0|[125]0{2,})$")
PENDING = "pending"
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


# =================================================================================================
#  Loading
# =================================================================================================
def load_axes(path: Path = AXES_FILE) -> dict:
    """Return the axes document, with a flat ``keys`` list in render order added for convenience."""
    axes = yaml.safe_load(path.read_text(encoding="utf-8"))
    axes["keys"] = [f"{g['key']}.{a['key']}" for g in axes["groups"] for a in g["axes"]]
    return axes


def load_registry(path: Path = REGISTRY_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def registered_tools(registry: dict) -> list[dict]:
    """Flatten the registry to tools in render order, each carrying its category."""
    return [{**tool, "category": category["key"]} for category in registry["categories"] for tool in category["tools"]]


def load_record(path: Path) -> tuple[dict, str] | None:
    """Split a record into its front-matter data and its markdown body.

    Returns None for a page that is not a record. The section holds ordinary pages too — its own
    landing page, for one — so carrying a `solver:` block is what makes a file a record rather than
    living in the right directory. A record whose block is missing or misspelled is then reported
    by the registry correspondence check, which knows what should have been there.
    """
    match = FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        return None
    front_matter = yaml.safe_load(match.group(1)) or {}
    if "solver" not in front_matter:
        return None
    return front_matter["solver"], match.group(2)


def load_records(directory: Path = RECORDS_DIR) -> dict[str, tuple[dict, str]]:
    loaded = ((path.stem, load_record(path)) for path in sorted(directory.glob("*.md")))
    return {stem: record for stem, record in loaded if record is not None}


# =================================================================================================
#  Validation
# =================================================================================================
def normalize(text: str) -> str:
    """Collapse a note to the form used for near-duplicate detection."""
    stripped = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(stripped.split())


def iter_notes(record: dict):
    """Yield every (location, note) pair in a record — capability cells and metadata alike."""
    for axis_key, cell in (record.get("capabilities") or {}).items():
        if isinstance(cell, dict) and cell.get("note"):
            yield axis_key, cell["note"]
    for field, note in ((record.get("metadata") or {}).get("notes") or {}).items():
        yield f"metadata.{field}", note


def metadata_value(record: dict, key: str):
    """A metadata field, from the record's `metadata:` block or from its top level."""
    block = record.get("metadata") or {}
    return block[key] if key in block else record.get(key)


def check_note(location: str, note) -> list[str]:
    """A note must be a mapping carrying `text`.

    Checked rather than tolerated: without this a bare string reads as a note but loses its URL and
    its `distinct` flag silently, and a mapping without `text` raises mid-render instead of being
    reported with the record it came from.
    """
    if note is None:
        return []
    if not isinstance(note, dict):
        return [f"{location}: note is {type(note).__name__}, expected a mapping with `text`"]
    if not str(note.get("text") or "").strip():
        return [f"{location}: note has no `text`"]
    return []


def check_structure(
    axes: dict,
    registry: dict,
    records: dict,
    comparison_page: Path = COMPARISON_PAGE,
    definitions_page: Path = DEFINITIONS_PAGE,
) -> list[str]:
    """Return every structural problem found. An empty list means the data is well formed."""
    problems = check_comparison_include(comparison_page)
    problems += check_definitions_include(definitions_page)
    problems += check_definitions(axes)
    tools = registered_tools(registry)
    registered = {tool["key"] for tool in tools}

    for key in registered - set(records):
        problems.append(f"{key}: registered in solver_registry.yaml but has no record")
    for key in set(records) - registered:
        problems.append(f"{key}: has a record but is not in solver_registry.yaml")

    expected_axes = set(axes["keys"])
    for tool in tools:
        if tool["key"] not in records:
            continue
        record, body = records[tool["key"]]
        problems += check_record(tool, record, body, expected_axes, axes)
    return problems


def check_record(tool: dict, record: dict, body: str, expected_axes: set[str], axes: dict) -> list[str]:
    """Every rule a single record must satisfy, gathered from the four things a record declares."""
    key = tool["key"]
    return [
        *check_cells(key, record, expected_axes, set(axes["marks"])),
        *check_metadata(key, record, axes["metadata"]),
        *check_scale_cells(key, record, axes["scale_columns"]["columns"]),
        *check_include(tool, body),
    ]


def check_cells(key: str, record: dict, expected_axes: set[str], marks: set[str]) -> list[str]:
    """One cell per declared axis, no others, each with a mark from the vocabulary."""
    problems: list[str] = []
    cells = record.get("capabilities") or {}
    for missing in sorted(expected_axes - set(cells)):
        problems.append(f"{key}: no cell for axis `{missing}`")
    for unknown in sorted(set(cells) - expected_axes):
        problems.append(f"{key}: cell for unknown axis `{unknown}`")
    for axis_key, cell in sorted(cells.items()):
        mark = cell.get("mark") if isinstance(cell, dict) else None
        if mark not in marks:
            problems.append(f"{key}: `{axis_key}` has mark {mark!r}, expected one of {sorted(marks)}")
        problems += check_note(f"{key}: `{axis_key}`", (cell or {}).get("note"))
    return problems


def check_metadata(key: str, record: dict, axes_metadata: list[dict]) -> list[str]:
    """Every declared metadata field present, and every metadata note well formed."""
    problems = [
        f"{key}: no `{field['key']}` metadata"
        for field in axes_metadata
        if metadata_value(record, field["key"]) in (None, "")
    ]
    for field, note in ((record.get("metadata") or {}).get("notes") or {}).items():
        problems += check_note(f"{key}: metadata `{field}`", note)
    return problems


def check_scale_cells(key: str, record: dict, scale_columns: list[dict]) -> list[str]:
    """A record carries one cell per scaling column, no others, each `pending` or a 1-2-5 grid size."""
    problems: list[str] = []
    cells = record.get("scale") or {}
    expected = {column["key"] for column in scale_columns}
    for missing in sorted(expected - set(cells)):
        problems.append(f"{key}: no cell for scaling column `{missing}`")
    for unknown in sorted(set(cells) - expected):
        problems.append(f"{key}: cell for unknown scaling column `{unknown}`")
    for column_key in sorted(expected & set(cells)):
        value = cells[column_key]
        if value != PENDING and not GRID_VALUE.match(str(value)):
            problems.append(
                f"{key}: scaling column `{column_key}` is {value!r}, expected `pending` or a 1-2-5 grid size"
            )
    return problems


def check_definitions(axes: dict) -> list[str]:
    """Every column a table can show must carry the definition the definitions page prints."""
    named = [
        *((f"axis `{g['key']}.{a['key']}`", a) for g in axes["groups"] for a in g["axes"]),
        *((f"scaling column `{c['key']}`", c) for c in axes["scale_columns"]["columns"]),
        ("scale_columns", axes["scale_columns"]),
        *((f"metadata `{f['label']}`", f) for f in axes["metadata"]),
    ]
    return [f"axes: {name} has no definition" for name, spec in named if not (spec.get("definition") or "").strip()]


def check_include(tool: dict, body: str) -> list[str]:
    """A profiled record must pull in its generated table; an excluded one has no page to."""
    key = tool["key"]
    if not tool.get("profile", True):
        return []
    include = f'--8<-- "generated/features/{key}.md"'
    return [] if include in body else [f"{key}: page does not include its generated feature table"]


def check_comparison_include(page_path: Path) -> list[str]:
    """The comparison page must pull in its generated tables.

    The same reasoning as `check_include`, one level up: the page keeps its hand-written framing,
    so dropping the include leaves a page that still reads as complete while both tables are gone.
    """
    if not page_path.exists():
        return [f"{_display_path(page_path)}: comparison page is missing"]
    if COMPARISON_INCLUDE not in page_path.read_text(encoding="utf-8"):
        return [f"{_display_path(page_path)}: page does not include its generated comparison tables"]
    return []


def check_definitions_include(page_path: Path = DEFINITIONS_PAGE) -> list[str]:
    """The definitions page must pull in its generated body — the same rule as the comparison page."""
    if not page_path.exists():
        return [f"{_display_path(page_path)}: capability-definitions page is missing"]
    if DEFINITIONS_INCLUDE not in page_path.read_text(encoding="utf-8"):
        return [f"{_display_path(page_path)}: page does not include its generated definitions"]
    return []


def check_near_duplicate_notes(records: dict) -> list[str]:
    """Report notes that collide only after normalization — same thought, two footnotes."""
    seen: dict[str, tuple[str, str, str]] = {}
    problems: list[str] = []
    for key, (record, _body) in sorted(records.items()):
        for location, note in iter_notes(record):
            if note.get("distinct"):
                continue
            text, url = note["text"], note.get("url") or ""
            fingerprint = f"{normalize(text)}\n{url}"
            previous = seen.get(fingerprint)
            if previous and previous[2] != text:
                problems.append(
                    f"{key}: `{location}` reads almost exactly like {previous[0]}: `{previous[1]}`.\n"
                    f"    Make them identical so they share one footnote, reword one so the "
                    f"difference is real, or mark it `distinct: true`."
                )
            elif not previous:
                seen[fingerprint] = (key, location, text)
    return problems


# =================================================================================================
#  Rendering — the per-solver feature tables
# =================================================================================================
def render_feature_table(axes: dict, record: dict, key: str) -> str:
    """Render one tool's feature table: every axis, one row each, notes as numbered footnotes.

    The notes are footnoted rather than written into the cells because they repeat: a modelling
    solver gives the same answer for eleven capabilities, and eleven copies of the same paragraph
    force the reader to compare them word by word to discover they are identical. Deduplicating
    into footnotes says it once and marks the cells that share it.
    """
    marks = axes["marks"]
    notes = _NoteIndex(key)
    metadata_notes = (record.get("metadata") or {}).get("notes") or {}
    facts = [
        f"| {field['label']} | {_metadata_markup(field, metadata_value(record, field['key']))} "
        f"| {notes.reference(_metadata_note(field, metadata_notes))} |"
        for field in axes["metadata"]
    ]
    rows = []

    for group in axes["groups"]:
        for axis in group["axes"]:
            cell = record["capabilities"][f"{group['key']}.{axis['key']}"]
            label = f"{group['label']} · {axis['label']}"
            rows.append(f"| {label} | {_mark_markup(marks, cell['mark'])} | {notes.reference(cell.get('note'))} |")

    scales = axes["scale_columns"]
    for column in scales["columns"]:
        rows.append(f"| {scales['label']} · {column['label']} | {_scale_markup(record['scale'][column['key']])} | |")

    # Declaration order in the axes file, so a mark added there cannot render in the table while
    # going unmentioned in the legend.
    legend = " · ".join(f"{m['glyph']} {m['legend']}" for m in marks.values())
    lines = [
        "<!-- Generated by scripts/capability_data.py — do not edit. -->",
        "",
        "### At a glance",
        "",
        '<div class="solver-features" markdown>',
        "",
        "| | | |",
        "|---|---|---|",
        *facts,
        "",
        "</div>",
        "",
        "### Capabilities",
        "",
        f"Support: {legend}",
        "",
        # The wrapper is what lets the label column wrap: the docs theme sets `white-space: nowrap`
        # on table cells, which would give every capability label its full one-line width.
        '<div class="solver-features" markdown>',
        "",
        "| Capability | Support | Notes |",
        "|---|:---:|---|",
        *rows,
        "",
        "</div>",
    ]
    if notes.definitions:
        lines += ["", *notes.definitions]
    return "\n".join(lines) + "\n"


class _NoteIndex:
    """Assigns one footnote per distinct note, so cells sharing a note share its number."""

    def __init__(self, key: str):
        self.key = key
        self.numbers: dict[tuple[str, str], int] = {}
        self.definitions: list[str] = []

    def reference(self, note) -> str:
        """Return the markdown footnote reference for a note, defining it on first sight."""
        if not note:
            return ""
        text, url = note["text"], note.get("url") or ""
        fingerprint = (text, url)
        if fingerprint not in self.numbers:
            self.numbers[fingerprint] = len(self.numbers) + 1
            body = inline(text) + (f" [Source]({url})" if url else "")
            self.definitions.append(f"[^{self.key}-{self.numbers[fingerprint]}]: {body}")
        return f"[^{self.key}-{self.numbers[fingerprint]}]"


# =================================================================================================
#  Rendering — the comparison page
# =================================================================================================
def render_comparison(axes: dict, registry: dict, records: dict) -> str:
    """Render the comparison page's two tables and the footnotes they share.

    One footnote sequence serves both, because the two tables answer different questions about the
    same tool — what it can do, and what it costs to adopt — and a single fact can be the reason
    behind an entry in either. Splitting the sequence would print such a fact twice.
    """
    notes = _NoteIndex("cmp")
    lines = [
        "<!-- Generated by scripts/capability_data.py — do not edit. -->",
        "",
        *_capability_grid(axes, registry, records, notes),
        "",
        *_metadata_table(axes, registry, records, notes),
    ]
    if notes.definitions:
        lines += ["", *notes.definitions]
    return "\n".join(lines) + "\n"


def _capability_grid(axes: dict, registry: dict, records: dict, notes: "_NoteIndex") -> list[str]:
    """The what-it-can-do half: one row per tool, one column per axis, marks in the shared glyphs.

    Written as HTML rather than as a markdown table, for the one thing markdown tables cannot
    express: a heading spanning the columns of a group, which is how the README table presents the
    same grid. The `markdown` attributes are what keep the cells' links and footnote references
    live inside the raw HTML.
    """
    marks = axes["marks"]
    scale_columns = axes["scale_columns"]["columns"]
    width = 1 + sum(len(group["axes"]) for group in axes["groups"]) + len(scale_columns)

    def tool_row(tool: dict, record: dict) -> str:
        cells = [
            f'<td{_edge(axis is group["axes"][0])} markdown="span">'
            f"{_mark_markup(marks, cell['mark'])}{notes.reference(cell.get('note'))}</td>"
            for group in axes["groups"]
            for axis in group["axes"]
            for cell in [record["capabilities"][f"{group['key']}.{axis['key']}"]]
        ]
        cells += [
            f'<td{_edge(column is scale_columns[0])} markdown="span">'
            f"{_scale_markup(record['scale'][column['key']])}</td>"
            for column in scale_columns
        ]
        return f'<tr markdown="block"><td markdown="span">{_tool_label(tool)}</td>{"".join(cells)}</tr>'

    def category_row(category: dict) -> str:
        # The label is wrapped so it can be pinned alongside the tool column: the cell spans the
        # full width, so once the grid is scrolled sideways an unpinned label sits off-screen and
        # the row reads as an unexplained gap.
        label = f'<span class="row-group-label">{category["label"]}</span>'
        return f'<tr><th colspan="{width}" scope="rowgroup">{label}</th></tr>'

    band = ['<th rowspan="2">Tool</th>']
    band += [f'<th{_edge(True)} colspan="{len(group["axes"])}">{group["label"]}</th>' for group in axes["groups"]]
    band.append(f'<th{_edge(True)} colspan="{len(scale_columns)}">{axes["scale_columns"]["hero_label"]}</th>')
    axis_names = "".join(
        f"<th{_edge(axis is group['axes'][0])}>{axis['hero_label']}</th>"
        for group in axes["groups"]
        for axis in group["axes"]
    )
    axis_names += "".join(
        f"<th{_edge(column is scale_columns[0])}>{column['hero_label']}</th>" for column in scale_columns
    )

    legend = " · ".join(f"{m['glyph']} {m['legend']}" for m in marks.values())
    return [
        '<div class="comparison-grid" markdown>',
        "",
        '<table markdown="block">',
        "<thead>",
        f"<tr>{''.join(band)}</tr>",
        f"<tr>{axis_names}</tr>",
        "</thead>",
        '<tbody markdown="block">',
        *_tool_rows(registry, records, category_row, tool_row),
        "</tbody>",
        "</table>",
        "",
        "</div>",
        "",
        f"Support: {legend}",
    ]


def _metadata_table(axes: dict, registry: dict, records: dict, notes: "_NoteIndex") -> list[str]:
    """The what-it-costs-to-adopt half: the prose fields, for the tools in the same order."""
    fields = [field for field in axes["metadata"] if field.get("comparison", True)]

    def tool_row(tool: dict, record: dict) -> str:
        record_notes = (record.get("metadata") or {}).get("notes") or {}
        cells = [
            _metadata_markup(field, metadata_value(record, field["key"]))
            + notes.reference(_metadata_note(field, record_notes))
            for field in fields
        ]
        return _row([_tool_label(tool), *cells])

    labels = [field["label"] for field in fields]

    def category_row(category: dict) -> str:
        return _row([f"**{category['label']}**", *([""] * len(labels))])

    return [
        '<div class="comparison-meta" markdown>',
        "",
        _row(["Tool", *labels]),
        _row(["---"] * (len(labels) + 1)),
        *_tool_rows(registry, records, category_row, tool_row),
        "",
        "</div>",
        "",
        f"*Every fact above was verified on or after {_oldest_verification(registry, records)}; "
        f"each tool's own date is in the table.*",
    ]


def _tool_rows(registry: dict, records: dict, category_row, tool_row) -> list[str]:
    """Rows for every registered tool, each category announced by a row of its own.

    The categories are the comparison's main point — a flat ranking across an exact solver, a
    one-shot picker and an anytime optimizer would be misleading — so they are structure in the
    table rather than something the surrounding prose has to keep saying. The two tables render a
    row differently, so each supplies its own row builders; the ordering lives here.
    """
    rows = []
    for category in registry["categories"]:
        rows.append(category_row(category))
        for tool in category["tools"]:
            rows.append(tool_row(tool, records[tool["key"]][0]))
    return rows


def _tool_label(tool: dict) -> str:
    """A tool's name, linked to its profile where it has one and tagged when it is the subject.

    The tag is what lets the stylesheet tint the subject's whole row, in a markdown table as
    readily as in the generated HTML one — neither can carry a class on the row itself, but both
    can be selected through a cell's content. Position is deliberately not the hook: it would tie
    the highlight to an ordering the registry is free to change.
    """
    name = f"[{tool['name']}](solvers/{tool['key']}.md)" if tool.get("profile", True) else tool["name"]
    return f'<span class="subject-name">{name}</span>' if tool.get("subject") else name


def _oldest_verification(registry: dict, records: dict) -> str:
    """The staleness of the whole table is the staleness of its least recently checked tool."""
    return min(str(metadata_value(records[tool["key"]][0], "verified")) for tool in registered_tools(registry))


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


# =================================================================================================
#  Rendering — the capability-definitions page
# =================================================================================================
def render_definitions(axes: dict) -> str:
    """Render the body of the capability-definitions page from the axes file's definition fields.

    Each table is wrapped in a `capability-definitions` div of its own: the docs theme keeps
    table cells on one line, and here both columns are prose that has to wrap — unlike the feature
    tables, whose second column is a mark and must not.
    """
    scales = axes["scale_columns"]
    lines = [
        "<!-- Generated by scripts/capability_data.py — do not edit. -->",
        "",
        "## Marks",
        "",
        "Every capability cell carries one of three marks. A mark composes with the column's "
        "definition: the mark states *how far* the tool meets the criterion the column claims.",
        "",
        '<div class="capability-definitions" markdown>',
        "",
        "| Mark | Meaning |",
        "|:---:|---|",
        *(
            f"| {_mark_markup(axes['marks'], name)} | {inline(mark['legend'])} |"
            for name, mark in axes["marks"].items()
        ),
        "",
        "</div>",
    ]
    # One section per hero group, in hero order, headed by the group's long label and naming each
    # column by its hero header — so a reader can go from any hero column to its definition without
    # translating names. The scaling section renders the same way, since the hero draws it as just
    # another group.
    for group in axes["groups"]:
        lines += [
            "",
            f"## {group['label']}",
            "",
            '<div class="capability-definitions" markdown>',
            "",
            "| Column | Definition |",
            "|---|---|",
            *(f"| {axis['hero_label']} | {inline(axis['definition'])} |" for axis in group["axes"]),
            "",
            "</div>",
        ]
    lines += [
        "",
        f"## {scales['label'].capitalize()}",
        "",
        inline(scales["definition"]),
        "",
        '<div class="capability-definitions" markdown>',
        "",
        "| Column | Definition |",
        "|---|---|",
        *(f"| {column['hero_label']} | {inline(column['definition'])} |" for column in scales["columns"]),
        "",
        "</div>",
        "",
        "Sizes print in suffix notation: k = 10³, M = 10⁶, B = 10⁹.",
        "",
        "## Tool facts",
        "",
        "These fields appear on each profile page and in the comparison page's companion table.",
        "",
        '<div class="capability-definitions" markdown>',
        "",
        "| Field | Definition |",
        "|---|---|",
        *(f"| {field['label']} | {inline(field['definition'])} |" for field in axes["metadata"]),
        "",
        "</div>",
    ]
    return "\n".join(lines) + "\n"


# =================================================================================================
#  Rendering — shared
# =================================================================================================
def _mark_markup(marks: dict, mark: str) -> str:
    """A mark's glyph, tagged with which mark it is so the stylesheet can weight the three apart.

    Without the tag a stylesheet cannot tell a dash from a tilde — CSS has no way to select on a
    cell's text — and the three marks are forced to share one treatment even though they do not
    carry equal weight: `not available` is the quietest claim of the three.
    """
    return f'<span class="mark mark-{mark}">{marks[mark]["glyph"]}</span>'


def _edge(starts_group: bool) -> str:
    """Mark the column that opens a group, so the rule between groups can be drawn in CSS.

    A markdown table would have to count columns for this; here the renderer already knows which
    cell is first in its group, so the fact travels with the cell instead of being re-derived.
    """
    return ' class="group-edge"' if starts_group else ""


def format_scale_value(n: int) -> str:
    """Return a grid size in the suffix notation the tables use: 500 -> `500`, 20000 -> `20k`.

    Grid values are 1, 2 or 5 times a power of ten, so the mantissa is always a whole number of
    the chosen unit and the formatting never rounds.
    """
    for divisor, suffix in ((10**9, "B"), (10**6, "M"), (10**3, "k")):
        if n >= divisor:
            return f"{n // divisor}{suffix}"
    return str(n)


def _scale_markup(value) -> str:
    """Render one scaling cell: a measured grid size in suffix notation, or the pending marker."""
    if value == PENDING:
        return '<span class="scale-pending">pending</span>'
    return f"n = {format_scale_value(int(value))}"


def _metadata_note(field: dict, record_notes: dict):
    """A field's note, unless this column opts out.

    One record field can back more than one column — the release is a version and a date — and the
    note explaining it belongs against one of them rather than repeated in both.
    """
    return record_notes.get(field["key"]) if field.get("note", True) else None


def _metadata_markup(field: dict, value) -> str:
    """Render one metadata value according to its declared kind."""
    kind = field.get("kind")
    if kind in ("release_version", "release_date"):
        part = (value or {}).get("version" if kind == "release_version" else "date")
        if part in (None, "none"):
            return "&mdash;"
        # Tagged so the cell can be kept off wrapping: these columns are narrow, and a date broken
        # across two lines reads as two numbers.
        return f'<span class="release-part">{part}</span>'
    if kind == "url":
        return f"[{value}]({value})"
    return inline(str(value))


def inline(text: str) -> str:
    """Collapse prose onto one line: a markdown table cell cannot contain a line break."""
    return " ".join((text or "").split())


# =================================================================================================
#  Entry point
# =================================================================================================
def check_fragments_are_current(
    axes: dict,
    registry: dict,
    records: dict,
    fragments_dir: Path = FRAGMENTS_DIR,
    comparison_fragment: Path = COMPARISON_FRAGMENT,
    definitions_fragment: Path = DEFINITIONS_FRAGMENT,
) -> list[str]:
    """Report committed fragments that no longer match what the records would render.

    Structural validity says nothing about this: editing a mark leaves the data perfectly well
    formed while the committed table still shows the old one. The test suite catches it, but only
    once the change reaches CI — this puts the same finding in front of whoever edited the record,
    at the moment they edited it.
    """
    problems: list[str] = []
    for tool in registered_tools(registry):
        if not tool.get("profile", True):
            continue
        key = tool["key"]
        if key not in records:
            continue
        path = fragments_dir / f"{key}.md"
        fresh = render_feature_table(axes, records[key][0], key)
        if not path.exists():
            problems.append(f"{key}: no generated feature table — run scripts/capability_data.py")
        elif path.read_text(encoding="utf-8") != fresh:
            problems.append(
                f"{key}: {_display_path(path)} is stale — the record has changed since it was "
                f"generated. Run scripts/capability_data.py."
            )

    if not comparison_fragment.exists():
        problems.append("comparison: no generated tables — run scripts/capability_data.py")
    elif comparison_fragment.read_text(encoding="utf-8") != render_comparison(axes, registry, records):
        problems.append(
            f"comparison: {_display_path(comparison_fragment)} is stale — a record has changed "
            f"since it was generated. Run scripts/capability_data.py."
        )

    if not definitions_fragment.exists():
        problems.append("definitions: no generated body — run scripts/capability_data.py")
    elif definitions_fragment.read_text(encoding="utf-8") != render_definitions(axes):
        problems.append(
            f"definitions: {_display_path(definitions_fragment)} is stale — the axes file has "
            f"changed since it was generated. Run scripts/capability_data.py."
        )
    return problems


def _display_path(path: Path) -> str:
    """Repo-relative where possible. Reporting a problem must never itself raise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def validate(
    axes: dict,
    registry: dict,
    records: dict,
    comparison_page: Path = COMPARISON_PAGE,
    definitions_page: Path = DEFINITIONS_PAGE,
) -> tuple[list[str], list[str]]:
    """Everything wrong with the data itself, split into the two severities.

    Deliberately excludes the fragment-drift check. Drift is a property of the *committed output*,
    not of the data, and writing is what resolves it — folding it in here would make the generator
    refuse to regenerate precisely when regeneration is what is needed, with an error telling the
    reader to run the command that just refused.
    """
    return (
        check_structure(axes, registry, records, comparison_page, definitions_page),
        check_near_duplicate_notes(records),
    )


def write_fragments(
    axes: dict,
    registry: dict,
    records: dict,
    fragments_dir: Path = FRAGMENTS_DIR,
    comparison_fragment: Path = COMPARISON_FRAGMENT,
    definitions_fragment: Path = DEFINITIONS_FRAGMENT,
) -> list[Path]:
    """Write one fragment per tool that has a profile page, plus the comparison page's tables and
    the capability-definitions body.

    Excluded tools get no feature table — they have no page for one to land in — but they are still
    rows in the comparison tables, which is the reason their records exist at all.
    """
    fragments_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tool in registered_tools(registry):
        if not tool.get("profile", True):
            continue
        record, _body = records[tool["key"]]
        path = fragments_dir / f"{tool['key']}.md"
        path.write_text(render_feature_table(axes, record, tool["key"]), encoding="utf-8")
        written.append(path)

    comparison_fragment.parent.mkdir(parents=True, exist_ok=True)
    comparison_fragment.write_text(render_comparison(axes, registry, records), encoding="utf-8")
    written.append(comparison_fragment)
    definitions_fragment.write_text(render_definitions(axes), encoding="utf-8")
    written.append(definitions_fragment)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate and report only; write nothing")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root to operate on; defaults to this checkout. Lets the checks run against "
        "a copy instead of the working tree.",
    )
    args = parser.parse_args()

    axes = load_axes(args.root / "data" / "capability_axes.yaml")
    registry = load_registry(args.root / "data" / "solver_registry.yaml")
    records = load_records(args.root / "docs" / "benchmarks" / "third_party" / "solvers")
    fragments_dir = args.root / "generated" / "features"
    comparison_fragment = args.root / "generated" / "comparison.md"
    comparison_page = args.root / "docs" / "benchmarks" / "third_party" / "comparison.md"
    definitions_fragment = args.root / "generated" / "capability_definitions.md"
    definitions_page = args.root / "docs" / "benchmarks" / "third_party" / "capability_definitions.md"
    structural, near_duplicates = validate(axes, registry, records, comparison_page, definitions_page)
    # Drift is only a problem in --check mode; in write mode the write is the fix. Rendering needs
    # sound data, so it is skipped when the structure is already broken.
    if args.check and not structural:
        structural += check_fragments_are_current(
            axes, registry, records, fragments_dir, comparison_fragment, definitions_fragment
        )

    for problem in structural:
        print(f"ERROR  {problem}", file=sys.stderr)
    for problem in near_duplicates:
        print(f"ERROR  {problem}", file=sys.stderr)
    if structural or near_duplicates:
        print(
            f"\n{len(structural) + len(near_duplicates)} problem(s) found in the capability data.",
            file=sys.stderr,
        )
        return 1

    if args.check:
        print(f"capability data OK: {len(records)} record(s), {len(axes['keys'])} axes")
        return 0

    written = write_fragments(axes, registry, records, fragments_dir, comparison_fragment, definitions_fragment)
    print(f"wrote {len(written)} fragment(s) under {_display_path(args.root / 'generated')}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
