from pathlib import Path

import pytest

from benchmarks.figures.style import TOOL_COLORS, save_webp, tool_color, tool_key
from benchmarks.solver_scaling.configs import CONFIGS


def test_every_scaling_tool_has_a_color():
    """Each tool with a scaling configuration resolves to a palette color."""
    # --- act / assert -----------------
    assert all(tool_color(c.tool) for c in CONFIGS)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("max-div[DEFAULT]", "max-div"),
        ("RDKit[MaxMinPicker]", "rdkit"),
        ("apricot[facility-location]", "apricot-select"),
        ("code-FDM[FairFlow]", "code-fdm"),
        ("random", "random"),
    ],
)
def test_tool_key_maps_record_labels_to_palette_keys(label: str, expected: str):
    """A record's `tool[variant]` label resolves to the key `TOOL_COLORS` uses for that tool."""
    # --- act / assert -----------------
    assert tool_key(label) == expected
    assert tool_color(expected) == TOOL_COLORS[expected]


def test_save_webp_writes_a_webp_file(tmp_path: Path):
    """The shared writer produces a webp file and creates missing parent folders."""
    # --- arrange ----------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "nested" / "figure.webp"

    # --- act --------------------------
    save_webp(fig, path)

    # --- assert -----------------------
    assert path.read_bytes()[8:12] == b"WEBP"
