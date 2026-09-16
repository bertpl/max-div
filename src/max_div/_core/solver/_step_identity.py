from dataclasses import dataclass


# =================================================================================================
#  SolverStepIdentity
# =================================================================================================
@dataclass(frozen=True)
class SolverStepIdentity:
    """Which solver step this is: its index and its name, with the total step count as context for the index."""

    step_index: int
    step_name: str
    n_steps: int

    def display_name(self) -> str:
        """Return the numbered display name, "step i/N - name"."""
        return f"step {self.step_index}/{self.n_steps} - {self.step_name}"
