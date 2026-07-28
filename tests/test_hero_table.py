"""Guards for the generated README hero table.

The committed SVGs are build products of `scripts/build_hero_table.py` and the capability records
it reads, so they are pinned against a fresh render: an edit to either the generator or a record
cannot land without the images following.

The rest of the file pins the *reading* of that data — which axes become columns, which name goes
in the gutter, which row is highlighted, and where the glyphs come from. Those are the joints where
the README could start saying something the documentation surfaces do not, and every one of them is
answered by the data rather than by this renderer. The data's own rules belong to
`scripts/capability_data.py` and are tested with it.
"""

import importlib.util
import re
import sys
from copy import deepcopy
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
def table(builder):
    """The committed capability data, as the hero reads it."""
    return builder.HeroTable.from_repo()


@pytest.fixture
def synthetic():
    """A two-tool triple exercising every fact the hero takes from the data.

    Deliberately unlike the committed data: an axis the hero does not show, a tool whose reference
    name is longer than its hero name, and a subject that is not the first row of the file.
    """
    axes = {
        "marks": {
            "full": {"hero_glyph": "✓", "legend": "built in"},
            "partial": {"hero_glyph": "~", "legend": "reachable"},
            "none": {"hero_glyph": "", "legend": "not available"},
        },
        "groups": [
            {
                "key": "distance",
                "hero_label": "distance",
                "axes": [
                    {"key": "l2", "hero_label": "L2", "hero": True},
                    {"key": "cosine", "hero_label": "cosine", "hero": True},
                    {"key": "internal", "hero_label": "internal", "hero": False},
                ],
            }
        ],
        "scale": {"hero_label": "max practical n"},
    }
    registry = {
        "categories": [
            {"key": "exact", "label": "Exact solvers", "tools": [{"key": "other", "name": "Other (Binding)"}]},
            {
                "key": "anytime",
                "label": "Anytime optimizers",
                "tools": [{"key": "own", "name": "own", "subject": True}],
            },
        ]
    }
    records = {
        "other": (
            {
                "capabilities": {
                    "distance.l2": {"mark": "partial"},
                    "distance.cosine": {"mark": "full"},
                    "distance.internal": {"mark": "full"},
                },
                "scale": {"max_practical_n": "3"},
            },
            "",
        ),
        "own": (
            {
                "capabilities": {
                    "distance.l2": {"mark": "full"},
                    "distance.cosine": {"mark": "none"},
                    "distance.internal": {"mark": "none"},
                },
                "scale": {"max_practical_n": "4-5"},
            },
            "",
        ),
    }
    return axes, registry, records


# =================================================================================================
#  Drift
# =================================================================================================
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_committed_svg_matches_fresh_render(builder, table, theme):
    # --- arrange -----------------------------------------
    committed = builder.OUT_DIR / f"hero_{theme}.svg"

    # --- act ---------------------------------------------
    rendered = builder.build_svg(table, theme)

    # --- assert ------------------------------------------
    assert committed.read_text(encoding="utf-8") == rendered, (
        f"docs/images/hero_{theme}.svg is stale — re-run scripts/build_hero_table.py"
    )


def test_render_is_deterministic(builder, table):
    # --- act ---------------------------------------------
    first = builder.build_svg(table, "light")
    second = builder.build_svg(table, "light")

    # --- assert ------------------------------------------
    assert first == second


def test_computed_geometry_is_integral(builder, table):
    """No computed coordinate may be a float.

    Determinism is the reason: a float formatted from platform-dependent arithmetic would make
    the committed bytes irreproducible and the drift test above flaky. Author-chosen literals
    (font sizes, stroke widths, opacities) are exempt — they cannot vary between runs.
    """
    # --- arrange -----------------------------------------
    geometry = ("x", "y", "x1", "y1", "x2", "y2", "width", "height", "dx", "dy", "points")

    # --- act ---------------------------------------------
    svg = builder.build_svg(table, "light")
    offenders = []
    for attr in geometry:
        for value in re.findall(rf'\b{attr}="([^"]+)"', svg):
            if "." in value:
                offenders.append(f"{attr}={value}")

    # --- assert ------------------------------------------
    assert not offenders, f"non-integer geometry would break reproducibility: {offenders}"


# =================================================================================================
#  What the hero takes from the capability data
# =================================================================================================
def test_columns_are_the_hero_visible_axes(builder, synthetic):
    """An axis the comparison page shows and the hero does not is a column here and not there."""
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic

    # --- act ---------------------------------------------
    hero = builder.HeroTable(axes, registry, records)

    # --- assert ------------------------------------------
    assert hero.groups == [("distance", [("L2", 1), ("cosine", 1)]), ("max practical n", [("", 2)])]
    assert [len(row["marks"]) for row in hero.rows] == [2, 2]


def test_rows_follow_the_registry_order_and_its_short_names(builder, synthetic):
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    registry["categories"][0]["tools"][0]["hero_name"] = "Other"

    # --- act ---------------------------------------------
    hero = builder.HeroTable(axes, registry, records)

    # --- assert ------------------------------------------
    assert hero.categories == ["Exact solvers", "Anytime optimizers"]
    assert [(row["name"], row["category"], row["scale"]) for row in hero.rows] == [("Other", 0, "3"), ("own", 1, "4-5")]


def test_the_highlighted_row_is_the_registry_subject(builder, synthetic):
    """The highlight follows the registry's `subject` flag.

    Not the first row and not a hard-coded name — either would tint the wrong tool the moment the
    registry is reordered or the package is renamed.
    """
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    moved = deepcopy(registry)
    moved["categories"][0]["tools"][0]["subject"] = True
    del moved["categories"][1]["tools"][0]["subject"]

    # --- act ---------------------------------------------
    svg = builder.build_svg(builder.HeroTable(axes, registry, records), "light")
    after = builder.build_svg(builder.HeroTable(axes, moved, records), "light")

    # --- assert ------------------------------------------
    assert 'font-weight="700">own<' in svg
    assert 'font-weight="400">Other (Binding)<' in svg
    assert 'font-weight="700">Other (Binding)<' in after
    assert 'font-weight="400">own<' in after


def test_glyphs_and_legend_wording_come_from_the_axes_file(builder, synthetic):
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    axes["marks"]["full"] = {"hero_glyph": "★", "legend": "shipped"}

    # --- act ---------------------------------------------
    svg = builder.build_svg(builder.HeroTable(axes, registry, records), "light")

    # --- assert ------------------------------------------
    assert "★" in svg
    assert "shipped" in svg
    assert "✓" not in svg
    assert "built in" not in svg


def test_a_mark_with_no_hero_glyph_leaves_its_cell_empty(builder, synthetic):
    """`none` is drawn as nothing here, which is what makes the grid readable at a glance."""
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    for record, _body in records.values():
        for cell in record["capabilities"].values():
            cell["mark"] = "none"

    # --- act ---------------------------------------------
    svg = builder.build_svg(builder.HeroTable(axes, registry, records), "light")

    # --- assert ------------------------------------------
    assert svg.count("✓") == 1, "the only check mark left should be the one in the legend"
    assert "not available" not in svg, "a mark the hero never draws has nothing to explain"
