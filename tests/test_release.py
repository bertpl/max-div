"""Guards for the release script's wait on the 'Push to Main' CI run (scripts/release.py)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "release.py"

_HEAD = "a" * 40


def _load_module():
    """Import the script by path — `scripts/` is maintainer tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location("release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load_module()


def _fake_gh(monkeypatch: pytest.MonkeyPatch, runs_for_commit: list[dict], states: list[dict]) -> list[list[str]]:
    """Answer `gh run list` with `runs_for_commit` and each `gh run view` with the next of `states`.

    Returns the list that records every command the script ran.
    """
    commands: list[list[str]] = []
    remaining_states = list(states)

    def run_command(cmd: list[str], **kw: object) -> str:
        commands.append(cmd)
        if cmd[:3] == ["gh", "run", "list"]:
            return json.dumps(runs_for_commit)
        elif cmd[:3] == ["gh", "run", "view"]:
            return json.dumps(remaining_states.pop(0))
        else:
            raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(_mod, "run_command", run_command)
    monkeypatch.setattr(_mod.time, "sleep", lambda _: None)
    return commands


def test_an_in_flight_run_is_polled_by_its_id_until_it_succeeds(monkeypatch: pytest.MonkeyPatch):
    """The run is looked up once by commit, then re-checked by its id, never by listing runs again."""
    # --- arrange ----------------------
    commands = _fake_gh(
        monkeypatch,
        runs_for_commit=[{"databaseId": 42}],
        states=[
            {"status": "queued", "conclusion": ""},
            {"status": "in_progress", "conclusion": ""},
            {"status": "completed", "conclusion": "success"},
        ],
    )

    # --- act --------------------------
    run_id = _mod._wait_for_main_ci(_HEAD)

    # --- assert -----------------------
    assert run_id == "42"
    list_commands = [cmd for cmd in commands if cmd[:3] == ["gh", "run", "list"]]
    assert len(list_commands) == 1
    assert list_commands[0][list_commands[0].index("--commit") + 1] == _HEAD
    assert [cmd[3] for cmd in commands if cmd[:3] == ["gh", "run", "view"]] == ["42", "42", "42"]


@pytest.mark.parametrize(
    "runs_for_commit, states",
    [
        pytest.param([], [], id="no_run_for_the_commit"),
        pytest.param([{"databaseId": 42}], [{"status": "completed", "conclusion": "failure"}], id="run_failed"),
    ],
)
def test_the_release_aborts_without_a_successful_run(
    monkeypatch: pytest.MonkeyPatch, runs_for_commit: list[dict], states: list[dict]
):
    """A commit with no run, or with a run that did not succeed, stops the release."""
    # --- arrange ----------------------
    _fake_gh(monkeypatch, runs_for_commit, states)

    # --- act / assert -----------------
    with pytest.raises(SystemExit):
        _mod._wait_for_main_ci(_HEAD)
