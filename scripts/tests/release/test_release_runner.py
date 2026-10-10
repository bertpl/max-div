"""These tests check the order of the release steps and how scripts/release/release_runner.py runs them."""

import subprocess

import pytest

from scripts.tests.release.helpers import load_release_module

_release_runner = load_release_module("release_runner")
_release_step = load_release_module("release_step")
_validation = load_release_module("phase_validation")
_release_commit = load_release_module("phase_release_commit")
_post_release = load_release_module("phase_post_release")

ReleasePhase = _release_step.ReleasePhase


class _RecordingStep(_release_step.ReleaseStep):
    """A release step that appends to `ran` when it runs or handles its failure, and raises `error` if one is given."""

    def __init__(self, phase: ReleasePhase, step_title: str, ran: list[str], error: BaseException | None = None):
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

    def on_failure(self, context: _release_step.ReleaseContext) -> None:
        self._ran.append(f"on_failure: {self._step_title}")


def test_the_release_steps_list_every_step_class_once():
    """`RELEASE_STEPS` holds 1 instance of every step class of the 3 phases."""
    # --- arrange ----------------------
    phase_bases = [_validation.ValidationStep, _release_commit.ReleaseCommitStep, _post_release.PostReleaseStep]
    step_classes = [cls for base in phase_bases for cls in base.__subclasses__()]

    # --- act --------------------------
    listed_classes = [type(step) for step in _release_runner.RELEASE_STEPS]

    # --- assert -----------------------
    assert sorted(cls.__name__ for cls in listed_classes) == sorted(cls.__name__ for cls in step_classes)


def test_the_release_steps_run_phase_by_phase():
    """The steps of each phase are contiguous, and the phases come in the order that `ReleasePhase` lists them."""
    # --- arrange ----------------------
    phase_order = list(ReleasePhase)

    # --- act --------------------------
    positions = [phase_order.index(step.phase) for step in _release_runner.RELEASE_STEPS]

    # --- assert -----------------------
    assert positions == sorted(positions)
    assert set(positions) == set(range(len(phase_order)))


def test_the_badge_metrics_are_gathered_before_the_release_commit_reads_them():
    """`GatherBadgeMetricsStep` comes before `StampAndCommitReleaseStep`, which stamps the README badges from it."""
    # --- act --------------------------
    step_types = [type(step) for step in _release_runner.RELEASE_STEPS]

    # --- assert -----------------------
    assert step_types.index(_validation.GatherBadgeMetricsStep) < step_types.index(
        _release_commit.StampAndCommitReleaseStep
    )


def test_run_release_numbers_the_steps_and_names_each_phase(capsys: pytest.CaptureFixture):
    """Each step is numbered by its position, and each phase's name is printed once, before its first step."""
    # --- arrange ----------------------
    ran: list[str] = []
    steps = [
        _RecordingStep(ReleasePhase.VALIDATION, "check a", ran),
        _RecordingStep(ReleasePhase.VALIDATION, "check b", ran),
        _RecordingStep(ReleasePhase.RELEASE_COMMIT, "commit c", ran),
    ]
    context = _release_step.ReleaseContext("1.2.3", badge_metrics=_release_step.BadgeMetrics(99.5, 10))

    # --- act --------------------------
    _release_runner.run_release(steps, context, is_dry_run=False)

    # --- assert -----------------------
    assert ran == ["check a", "check b", "commit c"]
    assert capsys.readouterr().out == (
        "\nValidation:\n  [ 1] check a\n  [ 2] check b\n\nRelease commit:\n  [ 3] commit c\n"
    )


def test_a_dry_run_runs_only_the_validation_steps(capsys: pytest.CaptureFixture):
    """A dry run runs only the validation steps, and reports that every precondition passed."""
    # --- arrange ----------------------
    ran: list[str] = []
    steps = [
        _RecordingStep(ReleasePhase.VALIDATION, "check a", ran),
        _RecordingStep(ReleasePhase.RELEASE_COMMIT, "commit b", ran),
        _RecordingStep(ReleasePhase.POST_RELEASE, "push c", ran),
    ]
    context = _release_step.ReleaseContext("1.2.3", badge_metrics=_release_step.BadgeMetrics(99.5, 10))

    # --- act --------------------------
    _release_runner.run_release(steps, context, is_dry_run=True)

    # --- assert -----------------------
    assert ran == ["check a"]
    assert "Dry run: every precondition passed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(SystemExit(1), id="fail_with_message"),
        pytest.param(subprocess.CalledProcessError(1, ["git"]), id="command_failure"),
    ],
)
def test_a_failing_step_handles_its_failure_and_stops_the_release(error: BaseException):
    """The failing step's `on_failure` runs, the failure propagates, and no later step runs."""
    # --- arrange ----------------------
    ran: list[str] = []
    steps = [
        _RecordingStep(ReleasePhase.RELEASE_COMMIT, "failing step", ran, error=error),
        _RecordingStep(ReleasePhase.RELEASE_COMMIT, "never runs", ran),
    ]
    context = _release_step.ReleaseContext("1.2.3")

    # --- act --------------------------
    with pytest.raises(type(error)):
        _release_runner.run_release(steps, context, is_dry_run=False)

    # --- assert -----------------------
    assert ran == ["failing step", "on_failure: failing step"]
