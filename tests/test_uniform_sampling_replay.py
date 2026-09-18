"""The replay figure carries every frame as positions into one coordinate table and draws only the last frame for
the JavaScript-less render; a frame naming a wrong position would only show up as a wrong picture on the built site.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _load_module(name: str):
    """Import a script module by path — `scripts/` is maintainer tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def replay():
    """Return the module under test, loaded once, after the explorer module it imports."""
    _load_module("uniform_sampling_explorer")
    return _load_module("uniform_sampling_replay")


# Five population items, of which item 4 is never selected, so the coordinate table holds four.
VECTORS = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8], [0.3, 0.7], [0.6, 0.1]], dtype=np.float32)


@pytest.fixture
def fragment(replay) -> str:
    """Return a replay of three frames, the last one selecting items 1, 2 and 3."""
    frames = [
        replay.ReplayFrame(0.0, 0.10, [0, 1, 3]),
        replay.ReplayFrame(2.5, 0.15, [0, 1, 2]),
        replay.ReplayFrame(40.0, 0.2, [1, 2, 3]),
    ]
    return replay.replay_fragment(
        VECTORS, frames, n=5, k=3, objective_keys=("l2",), population_image_url="pop.webp", description="test"
    )


def test_fragment_is_a_replay_div_ending_in_a_newline(fragment):
    assert fragment.startswith('<div class="usx-figure usx-replay"')
    assert fragment.endswith("</div>\n")


def test_fragment_carries_the_table_of_every_selected_item_and_the_frames_as_positions(fragment):
    # --- act --------------------------
    points = json.loads(re.search(r"data-points='([^']*)'", fragment).group(1))
    frames = json.loads(re.search(r"data-frames='([^']*)'", fragment).group(1))

    # --- assert -----------------------
    assert points == [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8], [0.3, 0.7]]  # item 4 never selected
    assert frames == [[0.0, 0.1, [0, 1, 3]], [2.5, 0.15, [0, 1, 2]], [40.0, 0.2, [1, 2, 3]]]


def test_only_the_last_frame_is_drawn_for_the_static_render(fragment):
    # --- act --------------------------
    drawn = re.findall(r'<g class="usx-item" data-p="(\d+)">', fragment)

    # --- assert -----------------------
    assert [int(p) for p in drawn] == [1, 2, 3]
    assert fragment.count('<circle class="usx-dot"') == 3
    assert fragment.count('<line class="usx-rug') == 6
    assert 'class="usx-slider" min="0" max="2" value="2"' in fragment
    assert "frame 3/3 · 40.0 s · diversity 0.200000" in fragment


def test_a_replay_needs_a_frame(replay):
    with pytest.raises(ValueError, match="at least one frame"):
        replay.replay_fragment(VECTORS, [], n=5, k=3, objective_keys=("l2",), population_image_url="p", description="d")
