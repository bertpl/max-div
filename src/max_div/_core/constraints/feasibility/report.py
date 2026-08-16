"""`FeasibilityReport` presents a feasibility-pipeline result to a caller.

The report adds the two things the raw result lacks: the violation floor expressed on the
constraints-score scale the solver reports, and a rendering that states what was and was not proven.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .lagrangian import FeasibilityResult, FeasibilityStatus


@dataclass(frozen=True)
class FeasibilityReport:
    """Whether a problem's constraints can be satisfied, with the evidence behind the answer.

    `status`, `selection`, `violation`, `lam_min` and `lam_max` carry through from
    `FeasibilityResult`; see its docstring for what each holds, and `FeasibilityStatus` for what
    each verdict does and does not claim.  The multipliers are an infeasibility certificate only
    under `INFEASIBLE`.

    Attributes:
        violation_floor: a certified lower bound on the weighted violation of every possible
            selection.  Positive only under `INFEASIBLE`, where it is what makes the verdict a
            proof; zero otherwise.
        constraints_score_ceiling: `violation_floor` mapped onto the 0-1 scale the solver reports
            its constraints score on, so the best score an infeasible problem admits is readable
            without converting by hand.  Assumes the default linear violation penalty.
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
        """Return a one-paragraph rendering of the verdict and the numbers behind it."""
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
    def from_result(cls, result: FeasibilityResult, score_normalization: float) -> "FeasibilityReport":
        """Wrap a pipeline result, converting its floor onto the constraints-score scale.

        The caller supplies `score_normalization` — the constant the constraints score scales a
        violation by, derived from the constraints and `k` — because the constraints package owns
        that formula and this subpackage sits below it.
        """
        floor = max(result.bound, 0.0) if result.status is FeasibilityStatus.INFEASIBLE else 0.0
        return cls(
            status=result.status,
            selection=result.selection,
            violation=result.violation,
            violation_floor=floor,
            constraints_score_ceiling=1.0 - score_normalization * floor,
            lam_min=result.lam_min,
            lam_max=result.lam_max,
        )
