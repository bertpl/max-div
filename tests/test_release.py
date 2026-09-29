"""These tests check how the release script waits for the 'Push to Main' CI run (scripts/release.py)."""

import json
from pathlib import Path

import pytest

from tests.helpers import load_script

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "release.py"

_HEAD = "a" * 40


_release = load_script("release", SCRIPT)


def _fake_gh(monkeypatch: pytest.MonkeyPatch, runs_for_commit: list[dict], states: list[dict]) -> list[list[str]]:
    """Answer `gh run list` with `runs_for_commit` and each `gh run view` with the next entry of `states`.

    Also patches `time.sleep` in the script to return immediately, so the polling loop does not wait.

    Returns:
        The list of commands run by the script, in order.
    """
    commands: list[list[str]] = []
    remaining_states = list(states)

    def run_command(cmd: list[str], **kw: object) -> str:
        """Record `cmd` and return the canned JSON answer for it; fail on any other command."""
        commands.append(cmd)
        if cmd[:3] == ["gh", "run", "list"]:
            return json.dumps(runs_for_commit)
        elif cmd[:3] == ["gh", "run", "view"]:
            return json.dumps(remaining_states.pop(0))
        else:
            raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(_release, "run_command", run_command)
    monkeypatch.setattr(_release.time, "sleep", lambda _: None)
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
    run_id = _release._wait_for_main_ci(_HEAD)

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
        _release._wait_for_main_ci(_HEAD)
