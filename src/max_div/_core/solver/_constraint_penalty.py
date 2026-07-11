from __future__ import annotations

from enum import StrEnum


class ConstraintPenalty(StrEnum):
    """How the magnitude of a constraint violation is penalized in the feasibility score.

    Members
    -------

        - LINEAR:    Penalty proportional to the violation (default).
        - QUADRATIC: Penalty proportional to the square of the violation, pushing the solver harder
                     away from large individual constraint violations (relative to many small ones).
    """

    LINEAR = "linear"
    QUADRATIC = "quadratic"
