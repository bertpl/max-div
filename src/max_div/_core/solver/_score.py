from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from max_div._core.constraints._constraints import _np_con_total_violation, _np_con_total_weighted_violation
from max_div._core.metrics._diversity import DiversitySignalFamily

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from max_div._core.constraints import Constraint
    from max_div._core.metrics._diversity import DiversityMetric


# =================================================================================================
#  Score
# =================================================================================================
@dataclass(frozen=True, slots=True)
class Score:  # noqa: PLW1641 — value-semantics-only hot-path object; deliberately unhashable
    """Object representing the multi-component score of a selection.

    A selection is a final or intermediate solution to a max-div problem with fairness constraints.

    The different components have strict priorities in order of appearance:

                                    size > constraints > diversity > div_non_zero > div_fgm.

    Only in case of a tie in a lower-priority component, the next higher-priority component is considered for
    comparisons.

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
    def as_tuple(self, soft: float = 0.0, ignore_infeasible_diversity: bool = False) -> tuple[float, ...]:
        """Return score as tuple, in order of descending priority, such that tuple-comparison yields correct results.

        :param soft: Softness parameter in [0.0 ,1.0] indicating how soft constraints should be treated.
                      0.0 = hard constraints (i.e. constraints score is absolute higher prio than diversity)
                     >0.0 = soft constraints (i.e. diversity score is partly 'mixed into' constraints score)
                                    constraints_soft = constraints^(1-soft) * diversity^soft
        :param ignore_infeasible_diversity: If `True`, diversity is set to 0.0 if constraints are not fully satisfied.
        """
        if ignore_infeasible_diversity and self.constraints < 1.0:
            # set scores of diversity & tie-breakers to 0.0 in case of infeasible solution
            # (also, don't perform 'soft constraint' computation, since that also takes into account diversity)
            return self.size, self.constraints, 0.0, *[0.0 for tb in self.div_tie_breakers]
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
    def __lt__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self.as_tuple() < other.as_tuple()
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self.as_tuple() <= other.as_tuple()
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self.as_tuple() == other.as_tuple()
        return NotImplemented

    def __str__(self) -> str:
        return f"size={self.size:.4f} | constraints={self.constraints:.4f} | diversity={self.diversity:.4f}"


# =================================================================================================
#  ScoreGenerator
# =================================================================================================
def _con_norm_constant(max_violations: Sequence[int], con_weights: NDArray[np.float32], quadratic: bool) -> float:
    """Return the constraint-score normalization constant `1 / (1 + worst-case total violation)`.

    The worst-case total violation is computed with the *same* aggregation used for live scoring: each
    constraint's worst-case violation (`max_violationsᵢ`) is encoded as a pure shortfall (magnitude in
    column 0) and run through `_np_con_total_weighted_violation`. Reusing the accelerator this way keeps
    the normalization from ever drifting out of lockstep with the score under any weights / penalty mode.
    """
    worst_case_con_values = np.zeros((len(max_violations), 2), dtype=np.int32)
    worst_case_con_values[:, 0] = max_violations
    return 1.0 / (1.0 + float(_np_con_total_weighted_violation(worst_case_con_values, con_weights, quadratic)))


class ScoreGenerator:
    """Utility class to generate Score objects from core metrics & data structures.

    This allows repetitive, duplicate computations to be performed & cached at object instantiation.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        n: int | np.int32,
        k: int,
        diversity_metric: DiversityMetric,
        diversity_tie_breakers: list[DiversityMetric],
        constraints: list[Constraint],
        penalty_quadratic: bool = False,
    ) -> None:
        """Initialize the ScoreGenerator.

        :param n: (int | np.int32) number of vectors in the max-div problem.
        :param k: (int) The target selection size for the max-div problem.
        :param diversity_metric: (DiversityMetric) The diversity metric used to compute diversity scores.
        :param diversity_tie_breakers: (list[DiversityMetric]) The list of diversity tie-breaker metrics.
        :param constraints: (list[Constraint]) The list of constraints used in the max-div problem.
        :param penalty_quadratic: (bool) If True, penalize constraint violations quadratically instead of linearly.
        """
        # --- size score computation ------------
        self._n = n
        self._k = k
        self._size_c0 = 1 / (1 + k)
        self._size_c1 = 1 / (1 + n - k)

        # --- constraint score computation ------
        self._penalty_quadratic = penalty_quadratic
        self._con_weights = np.array([con.weight for con in constraints], dtype=np.float32)
        if len(constraints) > 0:
            max_con_violations = [
                max(
                    con.min_count,  # in case of minimal selection
                    min(k, len(con.int_set)) - con.max_count,  # in case of maximal selection
                    0,  # in case we can never violate this constraint
                )
                for con in constraints
            ]
            self._con_c = _con_norm_constant(max_con_violations, self._con_weights, penalty_quadratic)
        else:
            self._con_c = 0.0  # no constraints -> always perfect score
        # fast unweighted-linear aggregation applies only when all weights are 1 and penalization is linear
        self._use_fast_con_path = (not penalty_quadratic) and bool(np.all(self._con_weights == 1.0))

        # --- diversity & tie-breakers ----------
        # each metric is bound to its signal family here, once — compute_score then just picks the
        # matching signal array per metric, without any per-call family lookups
        self._diversity_metric = diversity_metric
        self._diversity_metric_family: DiversitySignalFamily = diversity_metric.signal_family
        self._diversity_tie_breakers = diversity_tie_breakers
        self._tie_breakers_with_family = [(tb, tb.signal_family) for tb in diversity_tie_breakers]

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
            penalty_quadratic=self._penalty_quadratic,
        )

    # -------------------------------------------------------------------------
    #  Score computation
    # -------------------------------------------------------------------------
    def compute_score(
        self,
        n_selected: int | np.int32,
        con_values: NDArray[np.int32],
        separation_signals: NDArray[np.float32] | None = None,
        mean_distance_signals: NDArray[np.float32] | None = None,
    ) -> Score:
        """Compute the multi-component Score of a selection.

        :param n_selected: (int | np.int32) current number of selected vectors.
        :param con_values: (np.ndarray[np.int32]) current constraint-bound status (m x 2 array).
        :param separation_signals: (np.ndarray[np.float32] | None) the selected vectors' separation-family
                                   signal values; may be omitted when no configured metric consumes them.
        :param mean_distance_signals: (np.ndarray[np.float32] | None) the selected vectors' mean-distance-family
                                      signal values; may be omitted when no configured metric consumes them.

        The caller must pass the signal array of every family that the configured metrics (main +
        tie-breakers) consume; families no metric consumes may be omitted.
        """
        # --- individual scores ---------------------------
        if n_selected <= self._k:
            size_score = 1.0 - self._size_c0 * (self._k - n_selected)
        else:
            size_score = 1.0 - self._size_c1 * (n_selected - self._k)

        # --- constraint score --------------------------- (fast unweighted-linear path, else general)
        if self._con_c == 0.0:
            con_score = 1.0  # no constraints -> perfect score
        elif self._use_fast_con_path:
            con_score = 1.0 - self._con_c * _np_con_total_violation(con_values)
        else:
            con_score = 1.0 - self._con_c * float(
                _np_con_total_weighted_violation(con_values, self._con_weights, self._penalty_quadratic)
            )

        # --- construct Score object ----------------------
        # each metric reads the signal array of the family it was bound to at construction
        main_signals = (
            separation_signals
            if self._diversity_metric_family == DiversitySignalFamily.SEPARATION
            else mean_distance_signals
        )
        return Score(
            size=size_score,
            constraints=con_score,
            diversity=float(self._diversity_metric.compute(main_signals)),  # ty: ignore[invalid-argument-type]  # caller passes required families; hot path, so no assert
            div_tie_breakers=tuple(
                float(
                    tb.compute(
                        separation_signals if family == DiversitySignalFamily.SEPARATION else mean_distance_signals  # ty: ignore[invalid-argument-type]  # caller passes required families; hot path, so no assert
                    )
                )
                for tb, family in self._tie_breakers_with_family
            ),
        )
