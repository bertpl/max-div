import pytest

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score
from max_div._core.solver._score_checkpoint import ScoreCheckpoint
from max_div._core.solver._solve_timeline import SolveTimeline
from max_div._core.solver._solver_step import SolverStepResult
from max_div._core.solver._step_identity import SolverStepIdentity


def _step_result(*checkpoint_times: tuple[float, int]) -> SolverStepResult:
    """Return a step result whose checkpoints are at the given `(seconds, iterations)` after the step's start."""
    return SolverStepResult(
        score_checkpoints=[
            ScoreCheckpoint(
                SolverStepIdentity(1, "step"),
                Elapsed(t_elapsed_sec=t, n_iterations=n),
                Score(size=1.0, constraints=1.0, diversities=(0.5,)),
            )
            for t, n in checkpoint_times
        ]
    )


def test_a_step_starts_at_the_real_time_since_the_solve_started(fake_clock):
    """A step starts on the solve-wide axis after the time between steps, with the earlier steps' iterations."""
    # --- arrange ----------------------
    timeline = SolveTimeline()
    fake_clock.advance(1.0)
    timeline.record_step_start()
    fake_clock.advance(1.5)
    timeline.record_step_result(_step_result((1.5, 20)))
    fake_clock.advance(3.0)  # this time falls outside every step timer

    # --- act --------------------------
    elapsed_before_step = timeline.record_step_start()

    # --- assert -----------------------
    assert elapsed_before_step == Elapsed(t_elapsed_sec=5.5, n_iterations=20)


def test_a_finished_step_keeps_its_duration_and_moves_its_checkpoints_onto_the_solve_wide_axis(fake_clock):
    """Each step keeps its own duration, and its checkpoints are shifted by the step's start on the solve-wide axis."""
    # --- arrange ----------------------
    timeline = SolveTimeline()
    fake_clock.advance(1.0)

    # --- act --------------------------
    timeline.record_step_start()
    timeline.record_step_result(_step_result((0.5, 10), (1.5, 20)))
    fake_clock.advance(4.0)
    timeline.record_step_start()
    timeline.record_step_result(_step_result((0.2, 5)))

    # --- assert -----------------------
    assert timeline.step_durations == [Elapsed(1.5, 20), Elapsed(0.2, 5)]
    assert [checkpoint.elapsed for checkpoint in timeline.checkpoints] == [
        Elapsed(1.5, 10),
        Elapsed(2.5, 20),
        Elapsed(5.2, 25),
    ]


def test_recording_a_result_without_a_step_start_raises():
    """A step result with no recorded start raises, so no checkpoint is shifted by the previous step's start."""
    # --- arrange ----------------------
    timeline = SolveTimeline()
    timeline.record_step_start()
    timeline.record_step_result(_step_result((0.5, 10)))

    # --- act / assert -----------------
    with pytest.raises(RuntimeError, match="needs a matching record_step_start"):
        timeline.record_step_result(_step_result((0.2, 5)))
