"""Guards for the solver capability data and its generator.

Two kinds of guard. The committed feature tables under `generated/` are build products, so they
are pinned against a fresh render exactly as the hero SVGs are. And every rule the schema check
claims to enforce gets a test that feeds it the malformed shape and asserts it is rejected — a
validator is only worth its failure cases, and a happy-path test would pass just as well against
a check that returned nothing at all.
"""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "capability_data.py"


def _load_module():
    """Import the generator by path — `scripts/` is maintainer tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location("capability_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cd():
    return _load_module()


@pytest.fixture(scope="module")
def real(cd):
    """The committed data, as the generator sees it."""
    return cd.load_axes(), cd.load_registry(), cd.load_records()


@pytest.fixture
def synthetic(cd):
    """A minimal, valid axes/registry/records triple that each rejection test then breaks."""
    axes = {
        "groups": [
            {"key": "distance", "label": "distance metrics", "axes": [{"key": "l2", "label": "L2", "hero": True}]}
        ],
        "scale": {"key": "max_practical_n", "label": "largest practical problem size", "hero": True},
    }
    axes["keys"] = ["distance.l2"]
    registry = {"categories": [{"key": "c", "label": "C", "tools": [{"key": "tool", "name": "Tool"}]}]}
    record = {
        "name": "Tool",
        "scale": {"max_practical_n": "3", "rationale": "because"},
        "capabilities": {"distance.l2": {"mark": "full"}},
    }
    body = '--8<-- "generated/features/tool.md"'
    return axes, registry, {"tool": (record, body)}


def problems(cd, synthetic_triple):
    axes, registry, records = synthetic_triple
    return cd.check_structure(axes, registry, records)


# =================================================================================================
#  Drift — the committed fragments are build products
# =================================================================================================
def test_committed_fragments_match_a_fresh_render(cd, real):
    # --- arrange -----------------------------------------
    axes, registry, records = real

    # --- act / assert ------------------------------------
    for tool in cd.registered_tools(registry):
        if not tool.get("reference", True):
            continue
        record, _body = records[tool["key"]]
        committed = cd.FRAGMENTS_DIR / f"{tool['key']}.md"
        assert committed.read_text(encoding="utf-8") == cd.render_feature_table(axes, record, tool["key"]), (
            f"{committed.name} is stale — re-run scripts/capability_data.py"
        )


def test_excluded_tools_get_no_fragment(cd, real):
    """A record kept out of the reference has no page to include a table into."""
    # --- arrange -----------------------------------------
    _axes, registry, _records = real
    excluded = [t["key"] for t in cd.registered_tools(registry) if not t.get("reference", True)]

    # --- act / assert ------------------------------------
    assert excluded, "expected at least one record to be excluded from the reference"
    for key in excluded:
        assert not (cd.FRAGMENTS_DIR / f"{key}.md").exists()


# =================================================================================================
#  The committed data is well formed
# =================================================================================================
def test_committed_data_passes_both_checks(cd, real):
    # --- act ---------------------------------------------
    structural, near_duplicates = cd.validate(*real)

    # --- assert ------------------------------------------
    assert structural == []
    assert near_duplicates == []


# =================================================================================================
#  Structural rejections — one test per rule the check claims to enforce
# =================================================================================================
def test_unknown_axis_is_rejected(cd, synthetic):
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["capabilities"]["distance.made_up"] = {"mark": "full"}

    # --- act / assert ------------------------------------
    assert any("unknown axis" in p for p in problems(cd, synthetic))


def test_missing_cell_is_rejected(cd, synthetic):
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["capabilities"].clear()

    # --- act / assert ------------------------------------
    assert any("no cell for axis" in p for p in problems(cd, synthetic))


@pytest.mark.parametrize("mark", ["yes", "no", True, False, None, "FULL"])
def test_mark_outside_the_vocabulary_is_rejected(cd, synthetic, mark):
    """`yes`/`no` matter specifically: YAML would hand them over as booleans."""
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["capabilities"]["distance.l2"] = {"mark": mark}

    # --- act / assert ------------------------------------
    assert any("expected one of" in p for p in problems(cd, synthetic))


def test_record_without_a_registry_entry_is_rejected(cd, synthetic):
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    records["stranger"] = deepcopy(records["tool"])

    # --- act / assert ------------------------------------
    assert any("not in solver_registry" in p for p in cd.check_structure(axes, registry, records))


def test_registry_entry_without_a_record_is_rejected(cd, synthetic):
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    registry["categories"][0]["tools"].append({"key": "ghost", "name": "Ghost"})

    # --- act / assert ------------------------------------
    assert any("has no record" in p for p in cd.check_structure(axes, registry, records))


def test_missing_scale_rationale_is_rejected(cd, synthetic):
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["scale"]["rationale"] = "   "

    # --- act / assert ------------------------------------
    assert any("no rationale" in p for p in problems(cd, synthetic))


@pytest.mark.parametrize("value", ["", "12", "1-2-3", "n/a", "10^3"])
def test_malformed_scale_value_is_rejected(cd, synthetic, value):
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["scale"]["max_practical_n"] = value

    # --- act / assert ------------------------------------
    assert any("power of ten" in p for p in problems(cd, synthetic))


@pytest.mark.parametrize("value", ["3", "4-5"])
def test_a_single_value_and_a_range_are_both_accepted(cd, synthetic, value):
    """max-div reports a range; SCIP reports one value. Both are legitimate."""
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["scale"]["max_practical_n"] = value

    # --- act / assert ------------------------------------
    assert problems(cd, synthetic) == []


def test_page_without_its_include_is_rejected(cd, synthetic):
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    record, _body = records["tool"]
    records["tool"] = (record, "prose with no include line")

    # --- act / assert ------------------------------------
    assert any("does not include its generated feature table" in p for p in cd.check_structure(axes, registry, records))


def test_an_excluded_record_needs_no_include(cd, synthetic):
    """A record that is not published has no page for the fragment to land in."""
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    registry["categories"][0]["tools"][0]["reference"] = False
    records["tool"] = (records["tool"][0], "no include here")

    # --- act / assert ------------------------------------
    assert cd.check_structure(axes, registry, records) == []


# =================================================================================================
#  Near-duplicate notes
# =================================================================================================
def _with_notes(records, first, second):
    record = records["tool"][0]
    record["capabilities"]["distance.l2"] = {"mark": "partial", "note": first}
    record["metadata"] = {"notes": {"license": second}}
    return records


def test_notes_differing_only_in_wording_are_rejected(cd, synthetic):
    # --- arrange -----------------------------------------
    records = _with_notes(
        synthetic[2],
        {"text": "Reachable, but you build the model."},
        {"text": "reachable but you build the model"},
    )

    # --- act / assert ------------------------------------
    assert any("reads almost exactly like" in p for p in cd.check_near_duplicate_notes(records))


def test_identical_notes_are_fine(cd, synthetic):
    """Byte-identical text is the intended way to share a footnote, not a problem to report."""
    # --- arrange -----------------------------------------
    shared = {"text": "Reachable, but you build the model."}
    records = _with_notes(synthetic[2], shared, dict(shared))

    # --- act / assert ------------------------------------
    assert cd.check_near_duplicate_notes(records) == []


def test_the_distinct_flag_suppresses_the_rejection(cd, synthetic):
    # --- arrange -----------------------------------------
    records = _with_notes(
        synthetic[2],
        {"text": "Reachable, but you build the model."},
        {"text": "reachable but you build the model", "distinct": True},
    )

    # --- act / assert ------------------------------------
    assert cd.check_near_duplicate_notes(records) == []


def test_the_same_text_under_different_urls_is_two_notes(cd, synthetic):
    """The dedupe key is the text and its evidence link together."""
    # --- arrange -----------------------------------------
    records = _with_notes(
        synthetic[2],
        {"text": "Same claim.", "url": "https://example.org/a"},
        {"text": "Same claim.", "url": "https://example.org/b"},
    )

    # --- act / assert ------------------------------------
    assert cd.check_near_duplicate_notes(records) == []
