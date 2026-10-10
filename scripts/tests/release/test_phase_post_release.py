"""These tests check the post-release steps (scripts/release/phase_post_release.py)."""

import pytest

from scripts.tests.release.helpers import load_release_module

_post_release = load_release_module("phase_post_release")
_step = load_release_module("step")


def test_add_unreleased_section_inserts_every_empty_category_above_the_newest_version(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A fresh `## Unreleased` section with all 6 empty categories goes above the newest version section."""
    # --- arrange ----------------------
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## 0.2.0 (2026-10-10)\n\n### Fixed\n- Fix a bug\n")
    monkeypatch.setattr(_post_release, "CHANGELOG", changelog)

    # --- act --------------------------
    _post_release.AddUnreleasedSectionStep().run(_step.ReleaseContext("0.2.0"))

    # --- assert -----------------------
    assert changelog.read_text() == (
        "# Changelog\n\n"
        "## Unreleased\n\n### Added\n\n### Changed\n\n### Deprecated\n\n### Removed\n\n### Fixed\n\n### Security\n\n"
        "## 0.2.0 (2026-10-10)\n\n### Fixed\n- Fix a bug\n"
    )


def test_a_post_release_failure_prints_how_to_undo_the_release_commit_and_tag(capsys: pytest.CaptureFixture):
    """A failing post-release step prints the commands that reset main to before the tag and delete the tag."""
    # --- arrange ----------------------
    context = _step.ReleaseContext("1.2.3")

    # --- act --------------------------
    with pytest.raises(SystemExit):
        _post_release.PushMainAndTagStep().on_failure(context)

    # --- assert -----------------------
    stderr = capsys.readouterr().err
    assert "git reset --hard v1.2.3~1" in stderr
    assert "git tag -d v1.2.3" in stderr
