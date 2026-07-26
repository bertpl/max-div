"""Guards for the generated README hero table.

The committed SVGs are build products of `scripts/build_hero_table.py` plus its companion data
file. These tests pin two things: that the committed bytes still match a fresh render (so an
edit to either the generator or the data cannot land without the images following), and that
the data file itself is well formed.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_hero_table.py"


def _load_builder():
    """Import the generator by path — `scripts/` is maintainer tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location("build_hero_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def parsed(builder):
    return builder.parse_data(builder.DATA_FILE.read_text(encoding="utf-8"))


# =================================================================================================
#  Drift
# =================================================================================================
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_committed_svg_matches_fresh_render(builder, parsed, theme):
    # --- arrange -----------------------------------------
    categories, rows = parsed
    committed = builder.OUT_DIR / f"hero_{theme}.svg"

    # --- act ---------------------------------------------
    rendered = builder.build_svg(categories, rows, theme)

    # --- assert ------------------------------------------
    assert committed.read_text(encoding="utf-8") == rendered, (
        f"docs/images/hero_{theme}.svg is stale — re-run scripts/build_hero_table.py"
    )


def test_render_is_deterministic(builder, parsed):
    # --- arrange -----------------------------------------
    categories, rows = parsed

    # --- act ---------------------------------------------
    first = builder.build_svg(categories, rows, "light")
    second = builder.build_svg(categories, rows, "light")

    # --- assert ------------------------------------------
    assert first == second


def test_computed_geometry_is_integral(builder, parsed):
    """No computed coordinate may be a float.

    Determinism is the reason: a float formatted from platform-dependent arithmetic would make
    the committed bytes irreproducible and the drift test above flaky. Author-chosen literals
    (font sizes, stroke widths, opacities) are exempt — they cannot vary between runs.
    """
    # --- arrange -----------------------------------------
    categories, rows = parsed
    geometry = ("x", "y", "x1", "y1", "x2", "y2", "width", "height", "dx", "dy", "points")

    # --- act ---------------------------------------------
    svg = builder.build_svg(categories, rows, "light")
    offenders = []
    for attr in geometry:
        for value in re.findall(rf'\b{attr}="([^"]+)"', svg):
            if "." in value:
                offenders.append(f"{attr}={value}")

    # --- assert ------------------------------------------
    assert not offenders, f"non-integer geometry would break reproducibility: {offenders}"


# =================================================================================================
#  Companion data file
# =================================================================================================
def test_every_row_has_a_mark_per_column(builder, parsed):
    # --- arrange -----------------------------------------
    _categories, rows = parsed
    expected = sum(len(cols) for _, cols in builder.GROUPS) - 1  # the scale column is not a mark

    # --- act / assert ------------------------------------
    for row in rows:
        assert len(row["marks"]) == expected, row["name"]


def test_marks_use_only_the_documented_vocabulary(builder, parsed):
    # --- arrange -----------------------------------------
    _categories, rows = parsed

    # --- act / assert ------------------------------------
    for row in rows:
        unknown = set(row["marks"]) - {"Y", "~", "."}
        assert not unknown, f"{row['name']}: unknown mark(s) {sorted(unknown)}"


def test_every_row_cites_a_source(builder, parsed):
    """Each tool's marks must be defensible from one cited URL."""
    # --- arrange -----------------------------------------
    _categories, rows = parsed

    # --- act / assert ------------------------------------
    for row in rows:
        assert row["source"].startswith("https://"), f"{row['name']}: source is not an https URL"


def test_every_row_belongs_to_a_named_category(builder, parsed):
    # --- arrange -----------------------------------------
    categories, rows = parsed

    # --- act / assert ------------------------------------
    assert categories, "no categories declared"
    for row in rows:
        assert 0 <= row["category"] < len(categories), f"{row['name']}: row precedes its category header"


def test_scale_values_are_powers_of_ten(builder, parsed):
    # --- arrange -----------------------------------------
    _categories, rows = parsed

    # --- act / assert ------------------------------------
    for row in rows:
        assert re.fullmatch(r"\d(-\d)?", row["scale"]), f"{row['name']}: scale {row['scale']!r} is not `x` or `x-y`"


def test_max_div_is_present_and_first(builder, parsed):
    """The subject leads the table; the rendering also keys its highlight off this name."""
    # --- arrange -----------------------------------------
    _categories, rows = parsed

    # --- act / assert ------------------------------------
    assert rows[0]["name"] == "max-div"
