"""The Matplotlib style shared by every max-div figure, packaged so the `plot` extra ships it.

The style sits next to this module as `figure_style.mplstyle`. `figure_style` applies it for the span
of a `with` block, so a figure adopts it without touching the global rcParams; `figure_style_path`
hands the file to code that reads it directly (the docs figures, the mkdocs dpi hook).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.style

if TYPE_CHECKING:
    from collections.abc import Iterator

_STYLE_PATH = Path(__file__).resolve().parent / "figure_style.mplstyle"


def figure_style_path() -> Path:
    """Return the path to the packaged Matplotlib style sheet."""
    return _STYLE_PATH


@contextlib.contextmanager
def figure_style() -> Iterator[None]:
    """Apply the shared figure style for the span of the `with` block, leaving the global style untouched."""
    with matplotlib.style.context(_STYLE_PATH):
        yield
