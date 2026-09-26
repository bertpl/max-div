from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score
from max_div._core.solver._score_checkpoint import ScoreCheckpoint
from max_div._core.solver._solve_timeline import SolveTimeline
from max_div._core.solver._solver_step import SolverStepResult
from max_div._core.solver._step_identity import SolverStepIdentity


def _step_result(*elapsed: tuple[float, int]) -> SolverStepResult:
    """Return a step result whose checkpoints sit at the given `(seconds, iterations)` since the step's start."""
    return SolverStepResult(
        score_checkpoints=[
            ScoreCheckpoint(
                SolverStepIdentity(1, "step"),
                Elapsed(t_elapsed_sec=t, n_iterations=n),
                Score(size=1.0, constraints=1.0, diversities=(0.5,)),
            )
            for t, n in elapsed
        ]
    )


def test_a_step_starts_at_the_real_time_since_the_solve_started(fake_clock):
    """Time between steps counts on the axis, while the iteration count is the earlier steps' total."""
    # --- arrange ----------------------
    timeline = SolveTimeline()
    fake_clock.advance(1.0)
    timeline.start_step()
    fake_clock.advance(1.5)
    timeline.finish_step(_step_result((1.5, 20)))
    fake_clock.advance(3.0)  # time outside every step timer

    # --- act --------------------------
    elapsed_before_step = timeline.start_step()

    # --- assert -----------------------
    assert elapsed_before_step == Elapsed(t_elapsed_sec=5.5, n_iterations=20)


def test_a_finished_step_keeps_its_duration_and_moves_its_checkpoints_onto_the_solve_wide_axis(fake_clock):
    """Each step's duration stays its own, and its checkpoints move by where the step started."""
    # --- arrange ----------------------
    timeline = SolveTimeline()
    fake_clock.advance(1.0)

    # --- act --------------------------
    timeline.start_step()
    timeline.finish_step(_step_result((0.5, 10), (1.5, 20)))
    fake_clock.advance(4.0)
    timeline.start_step()
    timeline.finish_step(_step_result((0.2, 5)))

    # --- assert -----------------------
    assert timeline.step_durations == [Elapsed(1.5, 20), Elapsed(0.2, 5)]
    assert [checkpoint.elapsed for checkpoint in timeline.checkpoints] == [
        Elapsed(1.5, 10),
        Elapsed(2.5, 20),
        Elapsed(5.2, 25),
    ]
