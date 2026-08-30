"""This module defines the feasibility verdict types: the three-valued status and the result record around it."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


# =================================================================================================
#  FeasibilityStatus / FeasibilityResult
# =================================================================================================
class FeasibilityStatus(IntEnum):
    """`FeasibilityStatus` is the three-valued outcome of a feasibility analysis; the definite verdicts are proofs."""

    INFEASIBLE = -1
    UNKNOWN = 0
    FEASIBLE = 1


@dataclass(frozen=True)
class FeasibilityResult:
    """A `FeasibilityResult` records whether a constraint set can be satisfied, with the evidence behind the answer.

    Two of the result's numbers describe different things and are easy to conflate:

    - `violation_floor` is a property of the *problem* — no selection whatsoever can beat it, and
      none need attain it either;
    - `violation` and `violation_per_constraint` describe the *selection returned* — what it
      actually costs, and where.

    They relate as `violation_floor <= violation = sum of weight_i * violation_per_constraint[i]`.
    Equality proves `selection` is a minimum-violation selection, since it attains a bound nothing
    can beat; a gap proves nothing either way, because the bound may simply be loose.  `violation_floor` is
    a fractional dual value (`max(bound, 0)`), so compare the two with a tolerance, not `==`.

    Attributes:
        status: `FEASIBLE` (selection satisfies every constraint), `INFEASIBLE` (the multipliers
            prove none can), or `UNKNOWN` (nothing was established, in either direction).
        selection: the constructed selection of k item indices.  Always populated, under every
            status — an `UNKNOWN` verdict says nothing was *proven*, not that nothing was built,
            so a caller wanting a starting selection, not a verdict, can use it unchanged.
        violation: the total weighted violation of `selection` (0 when it satisfies everything).
        violation_per_constraint: how much `selection` misses each constraint by, in items, in
            constraint order.  Describes the selection, not the certified lower bound; weighted and summed it
            gives `violation`.  Being a real selection's profile, it is exact under both the
            linear and the quadratic violation penalty.
        bound: the best dual value reached.  `violation_floor` is its non-negative reading; the raw
            value is kept because a negative one still says how far the search got.
        lam_min: the shortfall multipliers behind `bound`; with `lam_max` they form the
            infeasibility certificate when `status` is `INFEASIBLE`, and nothing otherwise.
        lam_max: the excess multipliers behind `bound`.
    """

    status: FeasibilityStatus
    selection: NDArray[np.int64]
    violation: float
    violation_per_constraint: NDArray[np.int64]
    bound: float
    lam_min: NDArray[np.float64]
    lam_max: NDArray[np.float64]

    @property
    def violation_floor(self) -> float:
        """Return the certified lower bound on every possible selection's weighted violation.

        Zero unless infeasibility was proven — only then does the dual value bound anything.
        """
        return max(self.bound, 0.0) if self.status is FeasibilityStatus.INFEASIBLE else 0.0

    @property
    def is_certified(self) -> bool:
        """Return whether the verdict is one of the two proofs."""
        return self.status is not FeasibilityStatus.UNKNOWN

    def __str__(self) -> str:
        """Return a one-paragraph rendering of the verdict and the numbers behind it."""
        if self.status is FeasibilityStatus.FEASIBLE:
            return f"FEASIBLE: found a selection of {self.selection.shape[0]} items satisfying every constraint."
        if self.status is FeasibilityStatus.INFEASIBLE:
            return (
                f"INFEASIBLE: no selection can satisfy every constraint.  Every selection carries a total "
                f"weighted violation of at least {self.violation_floor:g}; the best selection found carries "
                f"{self.violation:g}."
            )
        return (
            f"UNKNOWN: neither a satisfying selection nor a proof that none exists was found — an "
            f"UNKNOWN verdict says nothing about whether the constraints can be satisfied.  The best "
            f"selection found carries a total weighted violation of {self.violation:g}."
        )
