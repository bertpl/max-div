"""The ways a scaling run can end, and how a run's record maps to one.

`TIMEOUT` and `MEMORY` are the resource limits the size sweep stops on. `SCALING_FAILURE` is a
non-resource refusal — a solver that cannot express the instance at this size (DPPy's k-DPP once
k exceeds the kernel's numerical rank). It is a real scaling limit, disclosed with its reason, but
it is neither a memory nor a time limit and carries no special-case logic.
"""

from enum import Enum

REASON_TIMEOUT = "timeout"  # parent-set: the run passed its hard-kill deadline
REASON_MEMORY = "memory"  # parent-set: the run's polled peak RSS crossed the memory cap


class Outcome(Enum):
    """A run's terminal state, read back from a record's `completed` / `reason` fields."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    SCALING_FAILURE = "scaling_failure"


def classify(completed: bool, reason: str | None) -> Outcome:
    """Map a run's `completed` flag and `reason` to its `Outcome`.

    A completed run is `SUCCESS`. A parent-killed run is `TIMEOUT` or `MEMORY` by its reason.
    Any other non-completion — the child raised before returning a valid selection — is a
    `SCALING_FAILURE`: the solver produced no answer here for a non-resource reason.
    """
    if completed:
        return Outcome.SUCCESS
    if reason == REASON_TIMEOUT:
        return Outcome.TIMEOUT
    if reason == REASON_MEMORY:
        return Outcome.MEMORY
    return Outcome.SCALING_FAILURE
