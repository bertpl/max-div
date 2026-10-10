"""These tests check the order of the release steps and how scripts/release/runner.py runs them."""

import subprocess

import pytest

from scripts.tests.release.helpers import load_release_module

_runner = load_release_module("runner")
_step = load_release_module("step")
_validation = load_release_module("phase_validation")
_release_commit = load_release_module("phase_release_commit")
_post_release = load_release_module("phase_post_release")

ReleasePhase = _step.ReleasePhase


class _RecordingStep(_step.ReleaseStep):
    """A _RecordingStep appends to `run_log` when it runs or handles its failure, and raises `error` if one is given."""

    def __init__(self, phase: ReleasePhase, step_title: str, run_log: list[str], error: BaseException | None = None):
        self.phase = phase
        self._step_title = step_title
        self._run_log = run_log
        self._error = error

    def title(self, context: _step.ReleaseContext) -> str:
        return self._step_title

    def run(self, context: _step.ReleaseContext) -> None:
        self._run_log.append(self._step_title)
        if self._error is not None:
            raise self._error

    def on_failure(self, context: _step.ReleaseContext) -> None:
        self._run_log.append(f"on_failure: {self._step_title}")


def test_the_release_steps_list_every_step_class_once():
    """`RELEASE_STEPS` holds 1 instance of every step class of the 3 phases."""
    # --- arrange ----------------------
    phase_bases = [
        _validation.ValidationPhaseStep,
        _release_commit.ReleaseCommitPhaseStep,
        _post_release.PostReleasePhaseStep,
    ]
    step_classes = [cls for base in phase_bases for cls in base.__subclasses__()]

    # --- act --------------------------
    listed_classes = [type(step) for step in _runner.RELEASE_STEPS]

    # --- assert -----------------------
    assert sorted(cls.__name__ for cls in listed_classes) == sorted(cls.__name__ for cls in step_classes)


def test_the_release_steps_run_phase_by_phase():
    """The steps of each phase are contiguous, and the phases come in the order that `ReleasePhase` lists them."""
    # --- arrange ----------------------
    phase_order = list(ReleasePhase)

    # --- act --------------------------
    positions = [phase_order.index(step.phase) for step in _runner.RELEASE_STEPS]

    # --- assert -----------------------
    assert positions == sorted(positions)
    assert set(positions) == set(range(len(phase_order)))


def test_the_badge_metrics_are_gathered_before_the_release_commit_reads_them():
    """`GatherBadgeMetricsStep` runs before the step that stamps the README badges from its metrics."""
    # --- act --------------------------
    step_types = [type(step) for step in _runner.RELEASE_STEPS]

    # --- assert -----------------------
    assert step_types.index(_validation.GatherBadgeMetricsStep) < step_types.index(
        _release_commit.StampReadmeAndSplashThenCommitStep
    )


def test_run_release_numbers_the_steps_and_names_each_phase(capsys: pytest.CaptureFixture):
    """Each step is numbered by its position, and each phase's name is printed once, before its first step."""
    # --- arrange ----------------------
    run_log: list[str] = []
    steps = [
        _RecordingStep(ReleasePhase.VALIDATION, "check a", run_log),
        _RecordingStep(ReleasePhase.VALIDATION, "check b", run_log),
        _RecordingStep(ReleasePhase.RELEASE_COMMIT, "commit c", run_log),
    ]
    context = _step.ReleaseContext("1.2.3", badge_metrics=_step.BadgeMetrics(99.5, 10))

    # --- act --------------------------
    _runner.run_release(steps, context, is_dry_run=False)

    # --- assert -----------------------
    assert run_log == ["check a", "check b", "commit c"]
    assert capsys.readouterr().out == (
        "\nValidation:\n  [ 1] check a\n  [ 2] check b\n\nRelease commit:\n  [ 3] commit c\n"
    )


def test_a_dry_run_runs_only_the_validation_steps(capsys: pytest.CaptureFixture):
    """A dry run runs only the validation steps, and reports that every precondition passed."""
    # --- arrange ----------------------
    run_log: list[str] = []
    steps = [
        _RecordingStep(ReleasePhase.VALIDATION, "check a", run_log),
        _RecordingStep(ReleasePhase.RELEASE_COMMIT, "commit b", run_log),
        _RecordingStep(ReleasePhase.POST_RELEASE, "push c", run_log),
    ]
    context = _step.ReleaseContext("1.2.3", badge_metrics=_step.BadgeMetrics(99.5, 10))

    # --- act --------------------------
    _runner.run_release(steps, context, is_dry_run=True)

    # --- assert -----------------------
    assert run_log == ["check a"]
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
    run_log: list[str] = []
    steps = [
        _RecordingStep(ReleasePhase.RELEASE_COMMIT, "failing step", run_log, error=error),
        _RecordingStep(ReleasePhase.RELEASE_COMMIT, "never runs", run_log),
    ]
    context = _step.ReleaseContext("1.2.3")

    # --- act --------------------------
    with pytest.raises(type(error)):
        _runner.run_release(steps, context, is_dry_run=False)

    # --- assert -----------------------
    assert run_log == ["failing step", "on_failure: failing step"]
