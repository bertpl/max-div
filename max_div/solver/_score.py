from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from max_div.sampling._constraint_helpers import _np_con_total_violation

from ._constraints import Constraint
from ._diversity import DiversityMetric


# =================================================================================================
#  Score
# =================================================================================================
@dataclass(frozen=True, slots=True)
class Score:
    """
    Object representing the multi-component score of a selection, i.e. of a final or intermediate solution to a
    max-div problem with fairness constraints.

    The different components have strict priorities in order of appearance:

                                    size > constraints > diversity > div_non_zero > div_fgm.

    Only in case of a tie in a lower-priority component, the next higher-priority component is considered for comparisons.

    All scores are >= 0.0, with higher being better.

    RATIONALE behind diversity tie-breakers:

      - these are optional additional metrics that can be added in case of ties in the main diversity score.

         - EX 1: min-dist only depends on the smallest distance.  Hence, swapping out any other vector
                 in the selection can have no effect on the diversity score, leading to many ties.
               --> Adding a tie-breaker such as geo-mean separation, can help pure swap-based algorithm converge
                   towards a more optimal solution.

         - EX 2: geo-mean will be 0.0 if any separation-value in the selection is 0.  If more than 1 such value is 0.0,
                 we have a situation where any single-vector swap will not affect the diversity score.
               --> Adding a tie-breaker that counts how many non-zero distances there are, helps guide the solver
                   towards removing vectors causing 0-distances.
    """

    # --- score components ------------
    size: float  # score indicating if target selection size is met
    constraints: float  # score indicating if constraints are satisfied
    diversity: float  # main diversity score, as computed by the user-selected diversity metric
    div_tie_breakers: tuple[float, ...]  # diversity tie-breakers - used in case of ties in all higher-prio metrics

    # --- helpers ---------------------
    def as_tuple(self, soft: float = 0.0) -> tuple[float, ...]:
        """
        Return score as tuple, in order of descending priority, such that tuple-comparison yields correct results.

        :param soft: Softness parameter in [0.0 ,1.0] indicating how soft constraints should be treated.
                      0.0 = hard constraints (i.e. constraints score is absolute higher prio than diversity)
                     >0.0 = soft constraints (i.e. diversity score is partly 'mixed into' constraints score)
                                    constraints_soft = constraints^(1-soft) * diversity^soft
        """
        if soft == 0.0:
            constraint_score = self.constraints  # 100% hard constraints (no influence from diversity)
        elif soft == 1.0:
            constraint_score = self.diversity  # 100% soft constraints (ignoring constraint score)
        elif self.constraints == 0.0 or self.diversity == 0.0:
            constraint_score = 0.0  # shortcut to avoid zero-division issues & unnecessary **soft computation
        else:
            constraint_score = self.constraints * ((self.diversity / self.constraints) ** soft)

        return self.size, constraint_score, self.diversity, *self.div_tie_breakers

    # --- math overloads --------------
    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Score):
            return self.as_tuple() < other.as_tuple()
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Score):
            return self.as_tuple() <= other.as_tuple()
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Score):
            return self.as_tuple() == other.as_tuple()
        return NotImplemented


# =================================================================================================
#  ScoreGenerator
# =================================================================================================
class ScoreGenerator:
    """
    Utility class to generate Score objects from core metrics & data structures, allowing repetitive, duplicate
    computations to be performed & cached at object instantiation.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        n: int,
        k: int,
        diversity_metric: DiversityMetric,
        diversity_tie_breakers: list[DiversityMetric],
        constraints: list[Constraint],
    ):
        """
        Initialize the ScoreGenerator.

        :param n: (int) number of vectors in the max-div problem.
        :param k: (int) The target selection size for the max-div problem.
        :param diversity_metric: (DiversityMetric) The diversity metric used to compute diversity scores.
        :param diversity_tie_breakers: (list[DiversityMetric]) The list of diversity tie-breaker metrics.
        :param constraints: (list[Constraint]) The list of constraints used in the max-div problem.
        """

        # --- size score computation ------------
        self._n = n
        self._k = k
        self._size_c0 = 1 / (1 + k)
        self._size_c1 = 1 / (1 + n - k)

        # --- constraint score computation ------
        if len(constraints) > 0:
            max_con_violations = [
                max(
                    con.min_count,  # in case of minimal selection
                    min(k, len(con.int_set)) - con.max_count,  # in case of maximal selection
                    0,  # in case we can never violate this constraint
                )
                for con in constraints
            ]
            self._con_c = 1 / (1 + sum(max_con_violations))
        else:
            self._con_c = 0.0  # no constraints -> always perfect score

        # --- diversity & tie-breakers ----------
        self._diversity_metric = diversity_metric
        self._diversity_tie_breakers = diversity_tie_breakers

        # --- store other params ---------------
        self._constraints = constraints

    # -------------------------------------------------------------------------
    #  Copy
    # -------------------------------------------------------------------------
    def copy(self) -> ScoreGenerator:
        """Create a deep copy of this ScoreGenerator."""
        return ScoreGenerator(
            n=self._n,
            k=self._k,
            diversity_metric=self._diversity_metric,
            diversity_tie_breakers=self._diversity_tie_breakers.copy(),
            constraints=self._constraints.copy(),
        )

    # -------------------------------------------------------------------------
    #  Score computation
    # -------------------------------------------------------------------------
    def compute_score(
        self, n_selected: int, con_values: NDArray[np.int32], selected_separation_array: NDArray[np.float32]
    ) -> Score:
        # --- individual scores ---------------------------
        if n_selected <= self._k:
            size_score = 1.0 - self._size_c0 * (self._k - n_selected)
        else:
            size_score = 1.0 - self._size_c1 * (n_selected - self._k)

        if self._con_c == 0.0:
            con_score = 1.0  # no constraints -> perfect score
        else:
            con_score = 1.0 - self._con_c * _np_con_total_violation(con_values)

        # --- construct Score object ----------------------
        return Score(
            size=size_score,
            constraints=con_score,
            diversity=float(self._diversity_metric.compute(selected_separation_array)),
            div_tie_breakers=tuple(float(tb.compute(selected_separation_array)) for tb in self._diversity_tie_breakers),
        )
