"""These tests check the changelog step of the post-release phase (scripts/release/post_release.py)."""

import pytest

from scripts.tests.helpers import load_release_module

_post_release = load_release_module("post_release")
_release_step = load_release_module("release_step")


def test_add_unreleased_section_inserts_every_empty_category_above_the_newest_version(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A fresh `## Unreleased` section with all 6 empty categories goes above the newest version section."""
    # --- arrange ----------------------
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## 0.2.0 (2026-10-10)\n\n### Fixed\n- Fix a bug\n")
    monkeypatch.setattr(_post_release, "CHANGELOG", changelog)

    # --- act --------------------------
    _post_release.AddUnreleasedSection().run(_release_step.ReleaseContext("0.2.0"))

    # --- assert -----------------------
    assert changelog.read_text() == (
        "# Changelog\n\n"
        "## Unreleased\n\n### Added\n\n### Changed\n\n### Deprecated\n\n### Removed\n\n### Fixed\n\n### Security\n\n"
        "## 0.2.0 (2026-10-10)\n\n### Fixed\n- Fix a bug\n"
    )
