"""The interactive figure of the geometric-mean distance guide precomputes its numbers in Python and carries
them in the fragment's `data-*` attributes; a wrong neighbor or a malformed fragment would only show up as
a wrong hover on the built site.
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE = REPO_ROOT / "scripts" / "geomean_distance_explorer.py"


def _load_module():
    """Import the module by path — `scripts/` is maintainer tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location("geomean_distance_explorer", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def explorer():
    """Return the module under test, loaded once."""
    return _load_module()


# The four items are chosen so that the distances disagree:
# - item 0 shares its y with item 1 and its x with item 2, so those pairs sit at distance 0 under the metric;
# - item 3 is nearest to item 2 under the metric (gaps 0.3 and 0.3) and under the Euclidean distance alike;
# - items 0 to 2 are Euclidean-nearest to item 3;
# - along x, items 0 and 2 share a value and item 3 ties between them at 0.3, so the first wins.
X = np.array([0.1, 0.9, 0.1, 0.4], dtype=np.float32)
Y = np.array([0.1, 0.1, 0.9, 0.6], dtype=np.float32)


# ==================================================================================================
#  Nearest neighbors
# ==================================================================================================
def test_nearest_neighbors_under_both_distances(explorer):
    """A shared coordinate gives distance 0 under the metric while the Euclidean neighbor is another item."""
    # --- act --------------------------
    neighbors = explorer.nearest_neighbors(X, Y)

    # --- assert -----------------------
    assert neighbors.index.tolist() == [1, 0, 0, 2]
    assert neighbors.distance[0] == 0.0
    assert neighbors.distance[3] == pytest.approx(0.3)
    assert neighbors.dx[3] == pytest.approx(0.3)
    assert neighbors.dy[3] == pytest.approx(0.3)
    assert neighbors.euclidean_index.tolist() == [3, 3, 3, 2]
    assert neighbors.x_index.tolist() == [2, 3, 0, 0]
    assert neighbors.y_index.tolist() == [1, 0, 3, 2]


# ==================================================================================================
#  Fragment
# ==================================================================================================
@pytest.fixture(scope="module")
def fragment(explorer):
    """Return the fragment of the four-item selection."""
    return explorer.explorer_fragment(X, Y, n=4, k=4, population_image="../images/pop.webp", description="four items")


def test_fragment_is_a_div_ending_in_a_newline(fragment):
    """A `<div>` root passes through Markdown untouched; the final newline keeps pre-commit from rewriting the file."""
    # --- assert -----------------------
    assert fragment.startswith('<div class="gmx-figure">\n')
    assert fragment.endswith("</div>\n")
    assert 'href="../images/pop.webp"' in fragment
    assert "<desc>four items</desc>" in fragment


def test_fragment_carries_one_dot_and_two_rug_ticks_per_item(fragment):
    """Every item has a dot, two rug ticks and two hit areas, all tagged with its index."""
    # --- act --------------------------
    dots = re.findall(r'<circle class="gmx-dot" data-i="(\d+)"', fragment)
    rugs = re.findall(r'<line class="gmx-rug gmx-rug-([xy])" data-i="(\d+)"', fragment)
    hits = re.findall(r'<line class="gmx-hit gmx-rug-([xy])" data-i="(\d+)"', fragment)

    # --- assert -----------------------
    assert dots == ["0", "1", "2", "3"]
    assert sorted(rugs) == sorted(hits) == sorted([(axis, str(i)) for axis in "xy" for i in range(4)])


def test_every_dot_names_its_neighbors_and_distance(fragment):
    """`geomean_distance_explorer.js` reads the `data-*` values on hover.

    They must index existing dots and match the arithmetic.
    """
    # --- act --------------------------
    dots = re.findall(
        r'<circle class="gmx-dot" data-i="(\d+)" data-nn="(\d+)" data-nne="(\d+)" data-nnx="(\d+)" data-nny="(\d+)"'
        r' data-d="([\d.]+)"',
        fragment,
    )

    # --- assert -----------------------
    assert [d[0] for d in dots] == ["0", "1", "2", "3"]
    assert all(0 <= int(j) < 4 for dot in dots for j in dot[1:5])
    assert dots[0][1:5] == ("1", "3", "2", "1")
    assert float(dots[0][5]) == 0.0
    assert float(dots[3][5]) == pytest.approx(0.3)
