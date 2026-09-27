from dataclasses import replace

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import WorkerGroupChange


def test_a_shifted_change_moves_only_its_elapsed():
    """Shifting adds the offset's time and iterations to `elapsed` and leaves every other field as it was."""
    # --- arrange ----------------------
    change = WorkerGroupChange(
        executed_by=1,
        elapsed=Elapsed(t_elapsed_sec=0.5, n_iterations=40),
        progress_fraction=0.3,
        dissolved_group=0,
        n_alive_groups_after=1,
        slot_scores={0: (1.0, 1.0, 0.2), 1: (1.0, 1.0, 0.4)},
        reassignments={0: 1},
    )

    # --- act --------------------------
    shifted = change.shifted_by(Elapsed(t_elapsed_sec=0.25, n_iterations=0))

    # --- assert -----------------------
    assert shifted == replace(change, elapsed=Elapsed(t_elapsed_sec=0.75, n_iterations=40))
