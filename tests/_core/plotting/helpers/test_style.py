import matplotlib

from tests._extras import skip_module_unless_extra

skip_module_unless_extra("plot")

from max_div._core.plotting.helpers import figure_style, figure_style_path  # noqa: E402


def test_the_style_sheet_ships_inside_the_package():
    """The style sheet sits next to the helper, so the `plot` extra carries it."""
    # --- arrange / act ----------------
    path = figure_style_path()

    # --- assert -----------------------
    assert path.name == "figure_style.mplstyle"
    assert path.is_file()


def test_the_style_applies_only_within_the_context():
    """`figure_style` sets the shared style for the block and restores the previous style after it."""
    # --- arrange ----------------------
    before = matplotlib.rcParams["font.family"]

    # --- act --------------------------
    with figure_style():
        inside = matplotlib.rcParams["font.family"]
    after = matplotlib.rcParams["font.family"]

    # --- assert -----------------------
    assert inside == ["Roboto"]
    assert after == before
