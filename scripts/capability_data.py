"""Load, validate and render the solver capability data.

Three kinds of file feed this, split by what they describe:

* ``data/capability_axes.yaml`` — the columns: which axes exist, their labels, hero visibility.
* ``data/solver_registry.yaml`` — the rows: categories in order, tools in order within each.
* ``docs/solvers/<key>.md`` — the cells and the prose. Each record's front matter is its data;
  its body is its reference page. One file, so a capability and the text defending it cannot
  drift apart.

Generated feature tables are written under ``generated/features/`` and pulled into each page by
a snippet line, which keeps the seam between hand-authored and generated content visible.

Two severities are reported. **Structural problems always fail**: an unknown axis, a missing
cell, a mark outside the vocabulary, a record that is not registered (or a registration with no
record), a missing scale rationale, a page that forgot its include. **Near-duplicate note text
also fails**, because two notes that differ only in wording produce two footnotes where one was
meant; a note may opt out with ``distinct: true`` when the resemblance really is coincidental.

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
RECORDS_DIR = REPO_ROOT / "docs" / "solvers"
FRAGMENTS_DIR = REPO_ROOT / "generated" / "features"

MARKS = {"full", "partial", "none"}
SCALE_PATTERN = re.compile(r"^\d(-\d)?$")
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
    return [
        {**tool, "category": category["key"], "category_label": category["label"]}
        for category in registry["categories"]
        for tool in category["tools"]
    ]


def load_record(path: Path) -> tuple[dict, str]:
    """Split a record into its front-matter data and its markdown body."""
    match = FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{path.name}: no YAML front matter")
    front_matter = yaml.safe_load(match.group(1))
    if "solver" not in front_matter:
        raise ValueError(f"{path.name}: front matter has no `solver:` key")
    return front_matter["solver"], match.group(2)


def load_records(directory: Path = RECORDS_DIR) -> dict[str, tuple[dict, str]]:
    return {path.stem: load_record(path) for path in sorted(directory.glob("*.md"))}


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


def check_structure(axes: dict, registry: dict, records: dict) -> list[str]:
    """Return every structural problem found. An empty list means the data is well formed."""
    problems: list[str] = []
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
        problems += check_record(tool, record, body, expected_axes)
    return problems


def check_record(tool: dict, record: dict, body: str, expected_axes: set[str]) -> list[str]:
    key = tool["key"]
    problems: list[str] = []

    cells = record.get("capabilities") or {}
    for missing in sorted(expected_axes - set(cells)):
        problems.append(f"{key}: no cell for axis `{missing}`")
    for unknown in sorted(set(cells) - expected_axes):
        problems.append(f"{key}: cell for unknown axis `{unknown}`")
    for axis_key, cell in sorted(cells.items()):
        mark = cell.get("mark") if isinstance(cell, dict) else None
        if mark not in MARKS:
            problems.append(f"{key}: `{axis_key}` has mark {mark!r}, expected one of {sorted(MARKS)}")

    scale = record.get("scale") or {}
    value = str(scale.get("max_practical_n", ""))
    if not SCALE_PATTERN.match(value):
        problems.append(f"{key}: scale {value!r} is not a power of ten or a range of them")
    if not (scale.get("rationale") or "").strip():
        problems.append(f"{key}: scale has no rationale")

    if tool.get("reference", True):
        include = f'--8<-- "generated/features/{key}.md"'
        if include not in body:
            problems.append(f"{key}: page does not include its generated feature table")

    return problems


def check_near_duplicate_notes(records: dict) -> list[str]:
    """Report notes that collide only after normalization — same thought, two footnotes."""
    seen: dict[str, tuple[str, str, str]] = {}
    problems: list[str] = []
    for key, (record, _body) in sorted(records.items()):
        for location, note in iter_notes(record):
            text = note["text"] if isinstance(note, dict) else str(note)
            if isinstance(note, dict) and note.get("distinct"):
                continue
            url = (note.get("url") or "") if isinstance(note, dict) else ""
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
#  Rendering
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
    rows = []

    for group in axes["groups"]:
        for axis in group["axes"]:
            cell = record["capabilities"][f"{group['key']}.{axis['key']}"]
            label = f"{group['label']} · {axis['label']}"
            rows.append(f"| {label} | {marks[cell['mark']]['glyph']} | {notes.reference(cell.get('note'))} |")

    scale = record["scale"]
    rows.append(
        f"| {axes['scale']['label']} | n ≈ 10<sup>{scale['max_practical_n']}</sup> "
        f"| {notes.reference({'text': scale['rationale']})} |"
    )

    legend = " · ".join(f"{marks[m]['glyph']} {marks[m]['legend']}" for m in ("full", "partial", "none"))
    lines = [
        "<!-- Generated by scripts/capability_data.py — do not edit. -->",
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
        text = note["text"] if isinstance(note, dict) else str(note)
        url = (note.get("url") or "") if isinstance(note, dict) else ""
        fingerprint = (text, url)
        if fingerprint not in self.numbers:
            self.numbers[fingerprint] = len(self.numbers) + 1
            body = inline(text) + (f" [Source]({url})" if url else "")
            self.definitions.append(f"[^{self.key}-{self.numbers[fingerprint]}]: {body}")
        return f"[^{self.key}-{self.numbers[fingerprint]}]"


def inline(text: str) -> str:
    """Collapse prose onto one line: a markdown table cell cannot contain a line break."""
    return " ".join((text or "").split())


# =================================================================================================
#  Entry point
# =================================================================================================
def check_fragments_are_current(axes: dict, registry: dict, records: dict) -> list[str]:
    """Report committed fragments that no longer match what the records would render.

    Structural validity says nothing about this: editing a mark leaves the data perfectly well
    formed while the committed table still shows the old one. The test suite catches it, but only
    once the change reaches CI — this puts the same finding in front of whoever edited the record,
    at the moment they edited it.
    """
    problems: list[str] = []
    for tool in registered_tools(registry):
        if not tool.get("reference", True):
            continue
        key = tool["key"]
        if key not in records:
            continue
        path = FRAGMENTS_DIR / f"{key}.md"
        fresh = render_feature_table(axes, records[key][0], key)
        if not path.exists():
            problems.append(f"{key}: no generated feature table — run scripts/capability_data.py")
        elif path.read_text(encoding="utf-8") != fresh:
            problems.append(
                f"{key}: {_display_path(path)} is stale — the record has changed since it was "
                f"generated. Run scripts/capability_data.py."
            )
    return problems


def _display_path(path: Path) -> str:
    """Repo-relative where possible. Reporting a problem must never itself raise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def validate(axes: dict, registry: dict, records: dict) -> tuple[list[str], list[str]]:
    structural = check_structure(axes, registry, records)
    # Only worth rendering once the data is sound; a malformed record would raise here rather than
    # report, and the structural problems are the ones to fix first anyway.
    if not structural:
        structural += check_fragments_are_current(axes, registry, records)
    return structural, check_near_duplicate_notes(records)


def write_fragments(axes: dict, registry: dict, records: dict) -> list[Path]:
    """Write one fragment per tool that has a reference page. Excluded tools get none."""
    FRAGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for tool in registered_tools(registry):
        if not tool.get("reference", True):
            continue
        record, _body = records[tool["key"]]
        path = FRAGMENTS_DIR / f"{tool['key']}.md"
        path.write_text(render_feature_table(axes, record, tool["key"]), encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate and report only; write nothing")
    args = parser.parse_args()

    axes, registry, records = load_axes(), load_registry(), load_records()
    structural, near_duplicates = validate(axes, registry, records)

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

    written = write_fragments(axes, registry, records)
    print(f"wrote {len(written)} feature table(s) to {FRAGMENTS_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
