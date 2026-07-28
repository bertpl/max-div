"""Guards for the solver capability data and its generator.

Two kinds of guard. The committed feature tables under `generated/` are build products, so they
are pinned against a fresh render exactly as the hero SVGs are. And every rule the schema check
claims to enforce gets a test that feeds it the malformed shape and asserts it is rejected — a
validator is only worth its failure cases, and a happy-path test would pass just as well against
a check that returned nothing at all.
"""

import importlib.util
import shutil
import subprocess
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
            {
                "key": "distance",
                "label": "distance metrics",
                "axes": [{"key": "l2", "label": "L2", "hero_label": "L2", "hero": True}],
            }
        ],
        "scale": {
            "key": "max_practical_n",
            "label": "largest practical problem size",
            "hero_label": "max practical n",
            "hero": True,
        },
        "metadata": [{"key": "guarantee", "label": "Guarantee"}],
        "marks": {
            "full": {"glyph": "Y", "legend": "built in"},
            "partial": {"glyph": "~", "legend": "reachable"},
            "none": {"glyph": "-", "legend": "not available"},
        },
    }
    axes["keys"] = ["distance.l2"]
    registry = {"categories": [{"key": "c", "label": "C", "tools": [{"key": "tool", "name": "Tool"}]}]}
    record = {
        "name": "Tool",
        "metadata": {"guarantee": "heuristic"},
        "scale": {"max_practical_n": "3", "rationale": "because"},
        "capabilities": {"distance.l2": {"mark": "full"}},
    }
    body = '--8<-- "generated/features/tool.md"'
    return axes, registry, {"tool": (record, body)}


def problems(cd, synthetic_triple):
    axes, registry, records = synthetic_triple
    # The committed comparison page carries its include, so passing it keeps these tests focused on
    # the record rule each one breaks. The include rule has its own tests below.
    return cd.check_structure(axes, registry, records, cd.COMPARISON_PAGE)


# =================================================================================================
#  Drift — the committed fragments are build products
# =================================================================================================
def test_committed_fragments_match_a_fresh_render(cd, real):
    # --- arrange -----------------------------------------
    axes, registry, records = real

    # --- act / assert ------------------------------------
    for tool in cd.registered_tools(registry):
        if not tool.get("profile", True):
            continue
        record, _body = records[tool["key"]]
        committed = cd.FRAGMENTS_DIR / f"{tool['key']}.md"
        assert committed.read_text(encoding="utf-8") == cd.render_feature_table(axes, record, tool["key"]), (
            f"{committed.name} is stale — re-run scripts/capability_data.py"
        )


def test_a_stale_fragment_is_reported_by_the_dry_run(cd, real, tmp_path):
    """The drift guard has to fire from `--check` too, not only from this suite.

    Otherwise the commit hook — which runs at the one moment the author still has the record in
    mind — reports everything fine while the committed table shows the previous marks.
    """
    # --- arrange -----------------------------------------
    axes, registry, records = real
    for tool in cd.registered_tools(registry):
        if tool.get("profile", True):
            (tmp_path / f"{tool['key']}.md").write_text("stale content", encoding="utf-8")

    # --- act ---------------------------------------------
    problems = cd.check_fragments_are_current(axes, registry, records, tmp_path)

    # --- assert ------------------------------------------
    assert any("is stale" in p for p in problems)


def test_a_missing_fragment_is_reported(cd, real, tmp_path):
    # --- arrange -----------------------------------------
    axes, registry, records = real

    # --- act ---------------------------------------------
    problems = cd.check_fragments_are_current(axes, registry, records, tmp_path)

    # --- assert ------------------------------------------
    assert any("no generated feature table" in p for p in problems)


def test_excluded_tools_get_no_fragment(cd, real):
    """A record kept out of the profiles has no page to include a table into."""
    # --- arrange -----------------------------------------
    _axes, registry, _records = real
    excluded = [t["key"] for t in cd.registered_tools(registry) if not t.get("profile", True)]

    # --- act / assert ------------------------------------
    assert excluded, "expected at least one record to be excluded from the profiles"
    for key in excluded:
        assert not (cd.FRAGMENTS_DIR / f"{key}.md").exists()


def test_the_committed_comparison_tables_match_a_fresh_render(cd, real):
    # --- arrange -----------------------------------------
    axes, registry, records = real

    # --- act / assert ------------------------------------
    assert cd.COMPARISON_FRAGMENT.read_text(encoding="utf-8") == cd.render_comparison(axes, registry, records), (
        "generated/comparison.md is stale — re-run scripts/capability_data.py"
    )


def test_a_stale_comparison_fragment_is_reported_by_the_dry_run(cd, real, tmp_path):
    # --- arrange -----------------------------------------
    axes, registry, records = real
    stale = tmp_path / "comparison.md"
    stale.write_text("stale content", encoding="utf-8")

    # --- act ---------------------------------------------
    problems = cd.check_fragments_are_current(axes, registry, records, tmp_path, stale)

    # --- assert ------------------------------------------
    assert any(p.startswith("comparison:") and "is stale" in p for p in problems)


def test_a_page_without_a_solver_block_is_not_a_record(cd, tmp_path):
    """The section holds ordinary pages too; carrying the block is what makes a file a record."""
    # --- arrange -----------------------------------------
    (tmp_path / "index.md").write_text("# Overview\n\nprose, no front matter\n", encoding="utf-8")
    (tmp_path / "other.md").write_text("---\ntitle: Something\n---\n\nprose\n", encoding="utf-8")

    # --- act ---------------------------------------------
    records = cd.load_records(tmp_path)

    # --- assert ------------------------------------------
    assert records == {}


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


def test_a_missing_metadata_field_is_rejected(cd, synthetic):
    """The declared fields are what the comparison table and each profile promise to show."""
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["metadata"] = {}

    # --- act / assert ------------------------------------
    assert any("no `guarantee` metadata" in p for p in problems(cd, synthetic))


def test_metadata_falls_back_to_the_record_top_level(cd, synthetic):
    """Source and verification date live at the top level, not inside the metadata block."""
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    axes["metadata"].append({"key": "verified", "label": "Last verified"})
    records["tool"][0]["verified"] = "2026-07-27"

    # --- act / assert ------------------------------------
    assert cd.check_structure(axes, registry, records) == []


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


@pytest.mark.parametrize("note", ["a bare string", 42, ["a", "list"]])
def test_a_note_that_is_not_a_mapping_is_rejected(cd, synthetic, note):
    """A bare string reads like a note but silently loses its URL and its `distinct` flag."""
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["capabilities"]["distance.l2"] = {"mark": "full", "note": note}

    # --- act / assert ------------------------------------
    assert any("expected a mapping" in p for p in problems(cd, synthetic))


def test_a_note_without_text_is_rejected(cd, synthetic):
    """Otherwise it raises mid-render, detached from the record that caused it."""
    # --- arrange -----------------------------------------
    synthetic[2]["tool"][0]["capabilities"]["distance.l2"] = {"mark": "full", "note": {"url": "https://x"}}

    # --- act / assert ------------------------------------
    assert any("no `text`" in p for p in problems(cd, synthetic))


def test_the_mark_vocabulary_comes_from_the_axes_file(cd, synthetic):
    """Validation and rendering must read one declaration, not two that can disagree."""
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    axes["marks"]["reachable"] = {"glyph": "?", "legend": "invented"}
    records["tool"][0]["capabilities"]["distance.l2"] = {"mark": "reachable"}

    # --- act / assert ------------------------------------
    assert cd.check_structure(axes, registry, records) == []


def test_page_without_its_include_is_rejected(cd, synthetic):
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    record, _body = records["tool"]
    records["tool"] = (record, "prose with no include line")

    # --- act / assert ------------------------------------
    assert any("does not include its generated feature table" in p for p in cd.check_structure(axes, registry, records))


def test_an_excluded_record_needs_no_include(cd, synthetic):
    """A record without a profile has no page for the fragment to land in."""
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    registry["categories"][0]["tools"][0]["profile"] = False
    records["tool"] = (records["tool"][0], "no include here")

    # --- act / assert ------------------------------------
    assert cd.check_structure(axes, registry, records) == []


# =================================================================================================
#  The comparison page
# =================================================================================================
def test_comparison_page_without_its_include_is_rejected(cd, tmp_path):
    # --- arrange -----------------------------------------
    page = tmp_path / "comparison.md"
    page.write_text("# Comparison\n\nprose, but no include\n", encoding="utf-8")

    # --- act / assert ------------------------------------
    assert any("does not include" in p for p in cd.check_comparison_include(page))


def test_a_missing_comparison_page_is_rejected(cd, tmp_path):
    """Reported rather than raised: the check runs from a commit hook, where a traceback is noise."""
    # --- act / assert ------------------------------------
    assert any("is missing" in p for p in cd.check_comparison_include(tmp_path / "gone.md"))


def test_every_registered_tool_is_a_row(cd, real):
    """Including the one kept out of the reference — needing a row here is why its record exists."""
    # --- arrange -----------------------------------------
    axes, registry, records = real

    # --- act ---------------------------------------------
    rendered = cd.render_comparison(axes, registry, records)

    # --- assert ------------------------------------------
    for tool in cd.registered_tools(registry):
        assert tool["name"] in rendered


def test_every_capability_axis_is_a_column(cd, real):
    """The bar this page is held to: it shows at least everything the README table shows."""
    # --- arrange -----------------------------------------
    axes, registry, records = real

    # --- act ---------------------------------------------
    rendered = cd.render_comparison(axes, registry, records)

    # --- assert ------------------------------------------
    for group in axes["groups"]:
        for axis in group["axes"]:
            assert f">{axis['hero_label']}</th>" in rendered
    assert f">{axes['scale']['hero_label']}</th>" in rendered


@pytest.mark.parametrize(
    "value, expected",
    [("3", "n ≈ 10<sup>3</sup>"), ("4-5", "n ≈ 10<sup>4</sup>&ndash;10<sup>5</sup>")],
)
def test_a_scale_range_renders_as_two_powers(cd, value, expected):
    """`10<sup>4-5</sup>` reads as a single exponent of `4-5` rather than as a range."""
    # --- act / assert ------------------------------------
    assert cd._scale_markup({"max_practical_n": value}) == expected


def test_each_group_heading_spans_its_own_columns(cd, real):
    """A markdown table cannot express this, which is why the grid is rendered as HTML.

    Asserted because a heading whose span drifts from its group silently mislabels every column
    to its right — the failure is invisible in the data and obvious only on the built page.
    """
    # --- arrange -----------------------------------------
    axes, registry, records = real

    # --- act ---------------------------------------------
    rendered = cd.render_comparison(axes, registry, records)

    # --- assert ------------------------------------------
    for group in axes["groups"]:
        assert f'colspan="{len(group["axes"])}">{group["label"]}</th>' in rendered


def test_a_metadata_field_can_be_kept_off_the_comparison_page(cd, real):
    """The source URL is reached through the link on each tool's name instead of its own column."""
    # --- arrange -----------------------------------------
    axes, registry, records = real
    hidden = [field for field in axes["metadata"] if not field.get("comparison", True)]

    # --- act ---------------------------------------------
    rendered = cd.render_comparison(axes, registry, records)
    per_solver = cd.render_feature_table(axes, records["scip"][0], "scip")

    # --- assert ------------------------------------------
    assert hidden, "expected at least one metadata field to be comparison-hidden"
    for field in hidden:
        assert f"| {field['label']} |" not in rendered
        assert f"| {field['label']} |" in per_solver


def test_a_note_shared_by_two_tools_becomes_one_footnote(cd, synthetic):
    """Dedupe runs across the whole page, which is the point of one footnote sequence for both
    tables — the same reason stated by two tools should not print twice."""
    # --- arrange -----------------------------------------
    axes, registry, records = synthetic
    shared = {"text": "Reachable, but you build the model."}
    records["tool"][0]["capabilities"]["distance.l2"] = {"mark": "partial", "note": shared}
    records["other"] = deepcopy(records["tool"])
    records["other"][0]["name"] = "Other"
    registry["categories"][0]["tools"].append({"key": "other", "name": "Other"})

    # --- act ---------------------------------------------
    rendered = cd.render_comparison(axes, registry, records)

    # --- assert ------------------------------------------
    assert rendered.count("[^cmp-1]:") == 1
    assert rendered.count("[^cmp-1]") == 3  # one definition, one reference per tool


def test_only_the_excluded_tool_is_unlinked(cd, real):
    """Every other name links to its profile; max-div has none among third-party tools."""
    # --- arrange -----------------------------------------
    _axes, registry, _records = real

    # --- act ---------------------------------------------
    rendered = cd.render_comparison(*real)

    # --- assert ------------------------------------------
    for tool in cd.registered_tools(registry):
        linked = f"[{tool['name']}](solvers/{tool['key']}.md)" in rendered
        assert linked is tool.get("profile", True)


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


# =================================================================================================
#  The command, not just the functions
# =================================================================================================
@pytest.fixture
def repo_copy(tmp_path):
    """A throwaway copy of everything the generator reads and writes.

    The subprocess tests need to make the data wrong on purpose, and the suite runs in parallel —
    so they work on a copy rather than the checkout, where a tampered file would be read by
    whatever other test happened to be running at the time.
    """
    for relative in ("data", "docs/solvers", "generated/features"):
        source = REPO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    for relative in ("generated/comparison.md", "docs/comparison.md"):
        shutil.copy(REPO_ROOT / relative, tmp_path / relative)
    return tmp_path


def _run(root, *args):
    """Invoke the generator the way the make target, the commit hook and CI all do."""
    return subprocess.run(  # noqa: S603 -- fixed, repo-local command
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        encoding="utf-8",
        check=False,  # the return code is what these tests assert on
    )


def test_the_check_command_passes_on_the_committed_data(repo_copy):
    # --- act ---------------------------------------------
    result = _run(repo_copy, "--check")

    # --- assert ------------------------------------------
    assert result.returncode == 0, result.stderr


def test_the_check_command_fails_on_a_stale_fragment(repo_copy):
    """Guards main()'s wiring, which no in-process test reaches.

    Every check is reachable as a function, so a regression in how the command reports or exits —
    a narrowed failure condition, a lost return code — would leave the suite green while the
    commit hook and CI wave the problem through.
    """
    # --- arrange -----------------------------------------
    fragment = repo_copy / "generated" / "features" / "scip.md"
    fragment.write_text(fragment.read_text(encoding="utf-8") + "\nstale\n", encoding="utf-8")

    # --- act ---------------------------------------------
    result = _run(repo_copy, "--check")

    # --- assert ------------------------------------------
    assert result.returncode == 1
    assert "is stale" in result.stderr


def test_the_check_command_fails_on_a_comparison_page_missing_its_include(repo_copy):
    """The include is the only thing tying the generated tables to the page that shows them."""
    # --- arrange -----------------------------------------
    page = repo_copy / "docs" / "comparison.md"
    page.write_text(page.read_text(encoding="utf-8").replace('--8<-- "generated/comparison.md"', ""), encoding="utf-8")

    # --- act ---------------------------------------------
    result = _run(repo_copy, "--check")

    # --- assert ------------------------------------------
    assert result.returncode == 1
    assert "does not include" in result.stderr


def test_the_write_command_regenerates_a_stale_fragment(repo_copy):
    """Writing is what resolves drift, so the write path must not refuse on it.

    Folding the drift check into shared validation made this command fail with an error telling
    the reader to run the command that had just refused.
    """
    # --- arrange -----------------------------------------
    fragment = repo_copy / "generated" / "features" / "scip.md"
    intended = fragment.read_text(encoding="utf-8")
    fragment.write_text("stale\n", encoding="utf-8")

    # --- act ---------------------------------------------
    result = _run(repo_copy)

    # --- assert ------------------------------------------
    assert result.returncode == 0, result.stderr
    assert fragment.read_text(encoding="utf-8") == intended
