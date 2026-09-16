from dataclasses import dataclass


# =================================================================================================
#  SolverStepIdentity
# =================================================================================================
@dataclass(frozen=True)
class SolverStepIdentity:
    """Which solver step this is: its index and its name."""

    step_index: int
    step_name: str
