"""Define the chart styling shared by every benchmark result figure: style sheet, tool colors, markers, webp output.

The solver-scaling charts and the head-to-head anytime curves plot the same tools, so one module owns a tool's color and the webp writer.
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
STYLE_SHEET = REPO_ROOT / "local" / "docs" / "figures" / "docs.mplstyle"

# `TOOL_COLORS` gives one color per tool, keyed by the tool's solver-registry key; the random
# baseline, which the registry does not list, is keyed by its record label. max-div, the subject of
# every comparison, is black; the third-party tools take the colors.
TOOL_COLORS: dict[str, str] = {
    "max-div": "#222222",  # black
    "ortools-cpsat": "#F29E4C",  # orange
    "scip": "#5DBB7A",  # green
    "highs": "#E8655F",  # coral
    "rdkit": "#9B7BD6",  # purple
    "fpsample": "#3FB8B0",  # teal
    "skmatter": "#E87EB4",  # pink
    "apricot-select": "#B0C24E",  # lime
    "qc-selector": "#C69A5B",  # tan
    "dppy": "#E9C34A",  # gold
    "code-fdm": "#7F8FA6",  # slate
    "kmedoids": "#4DBEDF",  # sky
    "random": "#A6A6A6",  # gray
}

# Color for a tool absent from `TOOL_COLORS`: tracked records may name tools no longer in the roster,
# and a chart must not fail on an unlisted name.
UNLISTED_TOOL_COLOR = "#B58AA5"  # mauve

# `MARKER_SHAPES` is cycled over when several series share a chart.
MARKER_SHAPES = ("o", "s", "D", "^", "v", "*", "p")

# Record labels read `tool[variant]`; the part before the bracket, lowercased, is the registry key
# for every tool but apricot, whose registry key carries the package name.
_TOOL_KEY_OVERRIDES = {"apricot": "apricot-select"}


def use_docs_style() -> None:
    """Activate the docs Matplotlib style sheet shared by all results figures."""
    plt.style.use(STYLE_SHEET)


def tool_color(tool: str) -> str:
    """Return the color of a tool by its `TOOL_COLORS` key, or `UNLISTED_TOOL_COLOR` for a tool not listed."""
    return TOOL_COLORS.get(tool, UNLISTED_TOOL_COLOR)


def tool_key(label: str) -> str:
    """Return the `TOOL_COLORS` key of a tool from a run record's `tool[variant]` label."""
    prefix = label.split("[", 1)[0]
    return _TOOL_KEY_OVERRIDES.get(prefix, prefix.lower())


def save_webp(fig: plt.Figure, path: Path) -> None:
    """Render the figure to lossy webp via PIL (matplotlib has no native webp writer) and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(buffer).convert("RGB").save(path, format="WEBP", lossless=False, quality=92)
    print(f"wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
