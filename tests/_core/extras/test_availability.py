import tomllib
from pathlib import Path

import pytest

from max_div._core.extras import (
    MissingExtraError,
    availability,
    declared_extras,
    is_extra_installed,
    missing_extra_message,
    require_extra,
)

_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _pyproject_extras() -> dict[str, list[str]]:
    with _PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def test_declared_extras_match_pyproject() -> None:
    assert declared_extras() == set(_pyproject_extras())


@pytest.mark.parametrize("extra", sorted(_pyproject_extras()))
def test_required_distributions_match_pyproject(extra: str) -> None:
    # --- arrange ----------------------
    expected = {availability._REQUIREMENT_NAME_PATTERN.match(req).group(1) for req in _pyproject_extras()[extra]}  # type: ignore[union-attr]

    # --- act --------------------------
    required = availability._required_distributions(extra)

    # --- assert -----------------------
    assert required == expected


def test_undeclared_extra_raises() -> None:
    with pytest.raises(ValueError, match="declares no extra 'no-such-extra'"):
        is_extra_installed("no-such-extra")


@pytest.mark.parametrize("every_one_present", [True, False])
def test_is_extra_installed_needs_every_distribution(monkeypatch: pytest.MonkeyPatch, every_one_present: bool) -> None:
    # --- arrange ----------------------
    monkeypatch.setattr(availability, "_required_distributions", lambda _: frozenset({"present", "other"}))
    monkeypatch.setattr(availability, "_is_installed", lambda name: name == "present" or every_one_present)

    # --- act / assert -----------------
    assert is_extra_installed("plot") is every_one_present


def test_is_installed_reads_the_environment() -> None:
    assert availability._is_installed("pytest")
    assert not availability._is_installed("no-such-distribution")


def test_require_extra_passes_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    # --- arrange ----------------------
    monkeypatch.setattr(availability, "is_extra_installed", lambda _: True)

    # --- act / assert -----------------
    require_extra("plot")


def test_require_extra_names_the_install_line_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # --- arrange ----------------------
    monkeypatch.setattr(availability, "is_extra_installed", lambda _: False)

    # --- act / assert -----------------
    with pytest.raises(MissingExtraError, match=r'pip install "max-div\[plot\]"'):
        require_extra("plot")


def test_missing_extra_message_names_the_extra() -> None:
    assert missing_extra_message("plot") == 'This feature requires the "plot" extra: pip install "max-div[plot]"'
