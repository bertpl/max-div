from pathlib import Path

import pytest

from benchmarks.figures.style import TOOL_COLORS, save_webp, tool_color, tool_key
from benchmarks.solver_scaling.configs import CONFIGS


def test_tool_colors_start_with_the_scaling_tools_in_configs_order():
    """The scaling tools keep the colors they had when the palette was indexed by CONFIGS order."""
    # --- arrange -----------------------------------------
    scaling_tools = list(dict.fromkeys(c.tool for c in CONFIGS))

    # --- assert ------------------------------------------
    assert list(TOOL_COLORS)[: len(scaling_tools)] == scaling_tools


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
def test_tool_key_maps_record_labels_to_registry_keys(label: str, expected: str):
    """A record's `tool[variant]` label resolves to the registry key the palette is keyed on."""
    # --- act / assert ------------------------------------
    assert tool_key(label) == expected
    assert tool_color(expected) == TOOL_COLORS[expected]


def test_save_webp_writes_a_webp_file(tmp_path: Path):
    """The shared writer produces a webp file and creates missing parent folders."""
    # --- arrange -----------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "nested" / "figure.webp"

    # --- act ---------------------------------------------
    save_webp(fig, path)

    # --- assert ------------------------------------------
    assert path.read_bytes()[8:12] == b"WEBP"
