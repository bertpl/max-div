"""These tests check the changelog step of the release commit (scripts/release/release_commit.py)."""

from datetime import date

import pytest

from scripts.tests.helpers import load_release_module

_release_commit = load_release_module("release_commit")
_release_step = load_release_module("release_step")


def test_finalize_changelog_dates_the_unreleased_section_and_drops_its_empty_categories(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The `## Unreleased` heading becomes the dated version heading, and categories without entries go."""
    # --- arrange ----------------------
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## Unreleased\n\n### Added\n\n### Fixed\n- Fix a bug\n\n### Security\n\n"
        "## 0.1.0 (2026-01-01)\n\n### Added\n- Add a feature\n"
    )
    monkeypatch.setattr(_release_commit, "CHANGELOG", changelog)

    # --- act --------------------------
    _release_commit.FinalizeChangelog().run(_release_step.ReleaseContext("0.2.0"))

    # --- assert -----------------------
    assert changelog.read_text() == (
        "# Changelog\n\n"
        f"## 0.2.0 ({date.today().isoformat()})\n\n### Fixed\n- Fix a bug\n\n"
        "## 0.1.0 (2026-01-01)\n\n### Added\n- Add a feature\n"
    )
