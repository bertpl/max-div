from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score
from max_div._core.solver._score_checkpoint import ScoreCheckpoint
from max_div._core.solver._step_identity import SolverStepIdentity


def test_a_shifted_checkpoint_moves_only_its_elapsed():
    """Shifting adds the offset's time and iterations to `elapsed` and leaves every other field as it was."""
    # --- arrange ----------------------
    checkpoint = ScoreCheckpoint(
        SolverStepIdentity(2, "step"),
        Elapsed(t_elapsed_sec=1.5, n_iterations=10),
        Score(size=1.0, constraints=1.0, diversities=(0.5,)),
        worker_index=3,
        group_index=1,
    )

    # --- act --------------------------
    shifted = checkpoint.shifted_by(Elapsed(t_elapsed_sec=2.0, n_iterations=5))

    # --- assert -----------------------
    assert shifted.elapsed == Elapsed(t_elapsed_sec=3.5, n_iterations=15)
    assert (shifted.step_identity, shifted.score, shifted.worker_index, shifted.group_index) == (
        checkpoint.step_identity,
        checkpoint.score,
        checkpoint.worker_index,
        checkpoint.group_index,
    )
