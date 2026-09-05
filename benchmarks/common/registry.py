"""Read the solver registry the documentation surfaces are built from, for the tools' display names."""

from pathlib import Path

import yaml

REGISTRY_FILE = Path(__file__).resolve().parents[2] / "data" / "solver_registry.yaml"


def solver_display_names() -> dict[str, str]:
    """Return the registry's tool key -> display name mapping."""
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {tool["key"]: tool["name"] for category in registry["categories"] for tool in category["tools"]}


def display_name(tool_key: str) -> str:
    """Return a tool's display name, or the key itself for a tool outside the registry."""
    return solver_display_names().get(tool_key, tool_key)
