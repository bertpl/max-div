import pytest

from benchmarks.solver_scaling.outcome import REASON_MEMORY, REASON_TIMEOUT, Outcome, classify


@pytest.mark.parametrize(
    ("completed", "reason", "expected"),
    [
        (True, None, Outcome.SUCCESS),
        (False, REASON_TIMEOUT, Outcome.TIMEOUT),
        (False, REASON_MEMORY, Outcome.MEMORY),
        (False, "ValueError: k=100 > rank=98", Outcome.SCALING_FAILURE),
        (False, None, Outcome.SCALING_FAILURE),
    ],
)
def test_classify_maps_completed_and_reason_to_the_outcome(completed, reason, expected):
    # --- act / assert -----------------
    assert classify(completed, reason) is expected


def test_a_completed_run_is_success_whatever_its_reason_field_holds():
    # A completed run's reason is None in practice, but completion wins regardless.
    # --- act / assert -----------------
    assert classify(True, REASON_TIMEOUT) is Outcome.SUCCESS
