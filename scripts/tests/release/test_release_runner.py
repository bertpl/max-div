"""These tests check the order of the release steps and how scripts/release/release_runner.py runs them."""

import subprocess

import pytest

from scripts.tests.helpers import load_release_module

_release_runner = load_release_module("release_runner")
_release_step = load_release_module("release_step")
_validation = load_release_module("validation")
_release_commit = load_release_module("release_commit")

Phase = _release_step.Phase


class _RecordingStep(_release_step.ReleaseStep):
    """A release step that appends its title to `ran` when it runs, and raises `error` if one is given."""

    def __init__(self, phase: Phase, step_title: str, ran: list[str], error: BaseException | None = None):
        self.phase = phase
        self._step_title = step_title
        self._ran = ran
        self._error = error

    def title(self, context: _release_step.ReleaseContext) -> str:
        return self._step_title

    def run(self, context: _release_step.ReleaseContext) -> None:
        self._ran.append(self._step_title)
        if self._error is not None:
            raise self._error


def _context() -> _release_step.ReleaseContext:
    """Return the context of a release of 1.2.3 whose badge metrics are already gathered."""
    return _release_step.ReleaseContext("1.2.3", badge_metrics=_release_step.BadgeMetrics(99.5, 10))


def test_the_release_steps_run_phase_by_phase():
    """The steps of each phase are contiguous, and the phases come in the order that `Phase` lists them."""
    # --- arrange ----------------------
    phase_order = list(Phase)

    # --- act --------------------------
    positions = [phase_order.index(step.phase) for step in _release_runner.RELEASE_STEPS]

    # --- assert -----------------------
    assert positions == sorted(positions)
    assert set(positions) == set(range(len(phase_order)))


def test_the_badge_metrics_are_gathered_before_the_release_commit_reads_them():
    """`GatherBadgeMetrics` comes before `CommitRelease`, which stamps the README badges from its result."""
    # --- act --------------------------
    step_types = [type(step) for step in _release_runner.RELEASE_STEPS]

    # --- assert -----------------------
    assert step_types.index(_validation.GatherBadgeMetrics) < step_types.index(_release_commit.CommitRelease)


def test_run_release_numbers_the_steps_and_names_each_phase(capsys: pytest.CaptureFixture):
    """Each step is numbered by its position, and each phase's name is printed once, before its first step."""
    # --- arrange ----------------------
    ran: list[str] = []
    steps = [
        _RecordingStep(Phase.VALIDATION, "check a", ran),
        _RecordingStep(Phase.VALIDATION, "check b", ran),
        _RecordingStep(Phase.RELEASE_COMMIT, "commit c", ran),
    ]

    # --- act --------------------------
    _release_runner.run_release(steps, _context(), is_dry_run=False)

    # --- assert -----------------------
    assert ran == ["check a", "check b", "commit c"]
    assert capsys.readouterr().out == (
        "\nValidation:\n  [ 1] check a\n  [ 2] check b\n\nRelease commit:\n  [ 3] commit c\n"
    )


def test_a_dry_run_runs_only_the_validation_steps(capsys: pytest.CaptureFixture):
    """A dry run stops before the first step that writes, and reports the badge metrics."""
    # --- arrange ----------------------
    ran: list[str] = []
    steps = [
        _RecordingStep(Phase.VALIDATION, "check a", ran),
        _RecordingStep(Phase.RELEASE_COMMIT, "commit b", ran),
        _RecordingStep(Phase.POST_RELEASE, "push c", ran),
    ]

    # --- act --------------------------
    _release_runner.run_release(steps, _context(), is_dry_run=True)

    # --- assert -----------------------
    assert ran == ["check a"]
    assert "Dry run: every precondition passed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "phase, error, is_recovery_hint_expected",
    [
        pytest.param(Phase.VALIDATION, SystemExit(1), False, id="validation"),
        pytest.param(Phase.RELEASE_COMMIT, SystemExit(1), False, id="release_commit"),
        pytest.param(Phase.POST_RELEASE, SystemExit(1), True, id="post_release"),
        pytest.param(
            Phase.POST_RELEASE, subprocess.CalledProcessError(1, ["git"]), True, id="post_release_command_failure"
        ),
    ],
)
def test_only_a_post_release_failure_prints_how_to_undo_the_release_commit_and_tag(
    capsys: pytest.CaptureFixture, phase: Phase, error: BaseException, is_recovery_hint_expected: bool
):
    """A failing step stops the release; only in the post-release phase do a local release commit and tag exist."""
    # --- arrange ----------------------
    ran: list[str] = []
    steps = [_RecordingStep(phase, "failing step", ran, error=error), _RecordingStep(phase, "never runs", ran)]

    # --- act --------------------------
    with pytest.raises(SystemExit):
        _release_runner.run_release(steps, _context(), is_dry_run=False)

    # --- assert -----------------------
    assert ran == ["failing step"]
    assert ("git reset --hard v1.2.3~1" in capsys.readouterr().err) == is_recovery_hint_expected
