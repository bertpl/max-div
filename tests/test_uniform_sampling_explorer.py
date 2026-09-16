"""The interactive figures of the uniform-sampling case study precompute their numbers in Python and carry
them in each fragment's `data-*` attributes; a wrong neighbor or a malformed fragment would only show up as
a wrong hover on the built site.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE = REPO_ROOT / "scripts" / "uniform_sampling_explorer.py"


def _load_module():
    """Import the module by path — `scripts/` is maintainer tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location("uniform_sampling_explorer", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def explorer():
    """Return the module under test, loaded once."""
    return _load_module()


# The four items are chosen so that the distances disagree:
# - item 0 shares its y with item 1 and its x with item 2, so those pairs sit at distance 0 under the
#   x or y distance, and hence under the L-inf and geometric-mean distances;
# - item 3 is nearest to item 2 under the geometric-mean distance (gaps 0.3 and 0.3) and under L2 alike;
# - items 0 to 2 are L2-nearest to item 3;
# - along x, items 0 and 2 share a value and item 3 ties between them at 0.3, so `argmin` picks the lower index, item 0.
X = np.array([0.1, 0.9, 0.1, 0.4], dtype=np.float32)
Y = np.array([0.1, 0.1, 0.9, 0.6], dtype=np.float32)


# ==================================================================================================
#  Nearest neighbors
# ==================================================================================================
def test_nearest_neighbors_under_each_distance(explorer):
    """A shared coordinate gives distance 0 under the product-like distances; the L2 neighbors are other items."""
    # --- act --------------------------
    neighbors = explorer.nearest_neighbors(X, Y, ("geomean", "linf", "l2", "x", "y"))

    # --- assert -----------------------
    assert neighbors["geomean"][0].tolist() == [1, 0, 0, 2]
    assert neighbors["geomean"][1][0] == 0.0
    assert neighbors["geomean"][1][3] == pytest.approx(0.3)
    assert neighbors["linf"][0].tolist() == [1, 0, 0, 2]
    assert neighbors["linf"][1][3] == pytest.approx(0.3)
    assert neighbors["l2"][0].tolist() == [3, 3, 3, 2]
    assert neighbors["l2"][1][3] == pytest.approx(0.3 * np.sqrt(2))
    assert neighbors["x"][0].tolist() == [2, 3, 0, 0]
    assert neighbors["y"][0].tolist() == [1, 0, 3, 2]


# ==================================================================================================
#  Fragment
# ==================================================================================================
@pytest.fixture(scope="module", params=[("geomean",), ("l2", "x", "y")], ids=["simple", "hybrid"])
def objective_keys(request):
    """Return the objective's distance keys: a simple objective and a hybrid."""
    return request.param


@pytest.fixture(scope="module")
def fragment(explorer, objective_keys):
    """Return the fragment of the four-item selection."""
    return explorer.explorer_fragment(
        X,
        Y,
        n=4,
        k=4,
        objective_keys=objective_keys,
        population_image_url="../images/pop.webp",
        description="four items",
    )


def test_fragment_is_a_div_ending_in_a_newline(fragment):
    """A `<div>` root passes through Markdown untouched; the final newline keeps pre-commit from rewriting the file."""
    # --- assert -----------------------
    assert fragment.startswith('<div class="usx-figure">\n')
    assert fragment.endswith("</div>\n")
    assert 'href="../images/pop.webp"' in fragment
    assert "<desc>four items</desc>" in fragment


def test_fragment_carries_one_dot_and_two_rug_ticks_per_item(fragment):
    """Every item has a dot, two rug ticks and two hit areas, all tagged with its index."""
    # --- act --------------------------
    dots = re.findall(r'<circle class="usx-dot" data-i="(\d+)"', fragment)
    rugs = re.findall(r'<line class="usx-rug usx-rug-([xy])" data-i="(\d+)"', fragment)
    hits = re.findall(r'<line class="usx-hit usx-rug-([xy])" data-i="(\d+)"', fragment)

    # --- assert -----------------------
    assert dots == ["0", "1", "2", "3"]
    assert sorted(rugs) == sorted(hits) == sorted([(axis, str(i)) for axis in "xy" for i in range(4)])


def test_every_dot_names_its_neighbors_under_the_objective_and_reference_distances(fragment, objective_keys):
    """`uniform_sampling_explorer.js` reads `data-nn` on hover: it must cover every objective and reference key."""
    # --- act --------------------------
    records = [
        json.loads(nn) for nn in re.findall(r"<circle class=\"usx-dot\" data-i=\"\d+\" data-nn='(\{.*?\})'", fragment)
    ]
    objective = re.search(r'data-objective="([^"]*)"', fragment).group(1)

    # --- assert -----------------------
    assert objective == " ".join(objective_keys)
    assert len(records) == 4
    assert all(set(record) == set(objective_keys) | {"l2", "x", "y"} for record in records)
    assert all(0 <= index < 4 for record in records for index, _ in record.values())
    assert records[0]["x"] == [2, 0.0]
    assert records[3]["l2"] == [2, pytest.approx(0.3 * np.sqrt(2), abs=1e-5)]


def test_legend_names_the_blue_neighbor_only_for_a_simple_objective(fragment, objective_keys):
    """A hybrid has no single nearest neighbor to paint blue; its per-term neighbors are the reference marks."""
    # --- assert -----------------------
    assert ("nearest neighbor, geometric-mean distance" in fragment) == (len(objective_keys) == 1)
    assert "nearest neighbor, L2 distance" in fragment


def test_band_edges_draw_one_line_per_edge_along_each_axis(explorer):
    """A constrained experiment shows its band edges; an unconstrained one draws no band line."""
    # --- act --------------------------
    banded = explorer.explorer_fragment(
        X, Y, n=4, k=4, objective_keys=("l2",), population_image_url="p.webp", description="d", band_edges=(0.5,)
    )
    plain = explorer.explorer_fragment(
        X, Y, n=4, k=4, objective_keys=("l2",), population_image_url="p.webp", description="d"
    )

    # --- assert -----------------------
    assert len(re.findall(r'<line class="usx-band"', banded)) == 2
    assert "usx-band" not in plain
