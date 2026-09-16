from dataclasses import dataclass


# =================================================================================================
#  SolverStepIdentity
# =================================================================================================
@dataclass(frozen=True)
class SolverStepIdentity:
    """Which solver step this is: its index, the step count that makes the index readable, and its name.

    Index 0 is the solver state initialization; the actual solver steps count from 1 to `n_steps`.
    """

    step_index: int
    n_steps: int
    step_name: str

    def display_name(self) -> str:
        """Return the numbered display name, "step i/N - name"."""
        return f"step {self.step_index}/{self.n_steps} - {self.step_name}"
