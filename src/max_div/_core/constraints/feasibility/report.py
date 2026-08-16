"""The user-facing answer to "can these constraints be satisfied at all?".

`FeasibilityReport` wraps the pipeline's raw outcome in the two things a caller needs on top of it:
the violation floor expressed on the constraints-score scale the solver reports, and a rendering
that states what was and was not proven.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints.constraints import Constraint, constraints_score_normalization

from .lagrangian import FeasibilityResult, FeasibilityStatus


@dataclass(frozen=True)
class FeasibilityReport:
    """Whether a problem's constraints can be satisfied, with the evidence behind the answer.

    Only the two definite verdicts carry information, and both are proofs.  `UNKNOWN` is not a
    weak "probably infeasible": it says the search settled nothing, and a caller must act exactly
    as if it had never run.

    Attributes:
        status: `FEASIBLE`, `INFEASIBLE`, or `UNKNOWN`.
        selection: a selection of `k` items — a witness satisfying every constraint when
            `FEASIBLE`; otherwise the least-violating one found, which under `UNKNOWN` comes with
            no claim of being the best possible.
        violation: the total weighted violation of `selection`, zero for a witness.
        violation_floor: a certified lower bound on the weighted violation of *every* possible
            selection.  Positive only when `INFEASIBLE`, where it is what makes the verdict a
            proof; zero otherwise.
        constraints_score_ceiling: `violation_floor` expressed on the 0-1 scale the solver reports
            its constraints score on, so an infeasible problem's best achievable score is visible
            without converting by hand.  Assumes the default linear violation penalty.
        lam_min: the shortfall multipliers behind the floor; with `lam_max` they are the
            infeasibility certificate, and `is_certified` says when they hold one.
        lam_max: the excess multipliers behind the floor.
    """

    status: FeasibilityStatus
    selection: NDArray[np.int64]
    violation: float
    violation_floor: float
    constraints_score_ceiling: float
    lam_min: NDArray[np.float64]
    lam_max: NDArray[np.float64]

    @property
    def is_certified(self) -> bool:
        """Return whether the verdict is one of the two proofs, rather than `UNKNOWN`."""
        return self.status is not FeasibilityStatus.UNKNOWN

    def __str__(self) -> str:
        if self.status is FeasibilityStatus.FEASIBLE:
            return f"FEASIBLE: found a selection of {self.selection.shape[0]} items satisfying every constraint."
        if self.status is FeasibilityStatus.INFEASIBLE:
            return (
                f"INFEASIBLE: no selection can satisfy every constraint.  Any selection violates them "
                f"by at least {self.violation_floor:g} (weighted), capping the constraints score at "
                f"{self.constraints_score_ceiling:.4f}.  The best selection found violates by "
                f"{self.violation:g}."
            )
        return (
            f"UNKNOWN: neither a satisfying selection nor a proof that none exists was found — this "
            f"says nothing about whether the constraints can be satisfied.  The best selection found "
            f"violates them by {self.violation:g} (weighted)."
        )

    @classmethod
    def from_result(cls, result: FeasibilityResult, constraints: list[Constraint], k: int) -> "FeasibilityReport":
        """Build a report from a pipeline result, converting the floor onto the score scale.

        Args:
            result: the pipeline outcome to wrap.
            constraints: the constraints it ran on, needed for the score conversion.
            k: the selection size, which sets the worst-case violation the conversion divides by.
        """
        floor = max(result.bound, 0.0) if result.status is FeasibilityStatus.INFEASIBLE else 0.0
        normalization = constraints_score_normalization(constraints, k)
        return cls(
            status=result.status,
            selection=result.selection,
            violation=result.violation,
            violation_floor=floor,
            constraints_score_ceiling=1.0 - normalization * floor,
            lam_min=result.lam_min,
            lam_max=result.lam_max,
        )
