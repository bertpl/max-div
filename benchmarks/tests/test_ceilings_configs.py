"""Guards for the per-tool configuration registry of the ceilings campaign."""

from pathlib import Path

import yaml

from benchmarks.ceilings.configs import TOOLS, Mode, resolve, seeds_for

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _registry_keys() -> set[str]:
    registry = yaml.safe_load((REPO_ROOT / "data" / "solver_registry.yaml").read_text(encoding="utf-8"))
    return {tool["key"] for category in registry["categories"] for tool in category["tools"]}


def test_every_registry_tool_has_a_campaign_entry_and_no_others() -> None:
    """The campaign covers exactly the published rows — a missing tool would silently keep
    its ceilings pending, and an extra one would measure something the table never shows."""
    # --- act / assert -----------------
    assert set(TOOLS) == _registry_keys()


def test_every_entry_carries_all_three_modes_with_descriptions() -> None:
    # --- act / assert -----------------
    for tool, entry in TOOLS.items():
        assert set(entry.configs) == set(Mode), tool
        for config in entry.configs.values():
            assert config.description.strip(), tool
        assert entry.memory_exponent in (1, 2), tool
        assert entry.memory_note.strip(), tool


def test_seed_counts_follow_the_protocol() -> None:
    """Five seeds for stochastic tools, three for deterministic-or-nearly ones."""
    # --- act / assert -----------------
    for tool, entry in TOOLS.items():
        assert len(seeds_for(tool)) == (5 if entry.stochastic else 3), tool


def test_the_sleep_fixture_resolves_but_is_not_a_registry_tool() -> None:
    """The kill-path test needs it; the published table must never see it."""
    # --- act / assert -----------------
    assert resolve("_test_sleep", Mode.FASTEST_VALID) is not None
    assert "_test_sleep" not in TOOLS
