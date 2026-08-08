"""Constraint-satisfaction scores for constrained sampling: from-scratch computation + incremental upkeep.

`_compute_score` scores every item in `[0, n)` from scratch — O(n + total constraint membership)
work, although a single draw changes only the drawn item's own constraints.

The `ConstraintScoreState` bundle removes that repetition: `new_constraint_score_state` computes
the scores once per sampling call, and `apply_draw` then updates only the drawn item's constraints
— plus, when one of them crosses a satisfaction threshold, that constraint's members.

The upkeep is exact — see `ConstraintScoreState` for the invariant.

The bundle is a namedtuple of numpy arrays, like `DistanceStore`, so it crosses the njit boundary
without object-mode.
"""

from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints._constraints import _np_con_indices, _np_con_max_value, _np_con_min_value

_SCORE_PENALTY_HARD_CONSTRAINT = np.int32(2**24)
_SCORE_PENALTY_ALREADY_SAMPLED = np.int32(2**30)

# Floor that `_compute_score` clamps each score to after a max-count penalty.
_SCORE_CLAMP_FLOOR = -_SCORE_PENALTY_ALREADY_SAMPLED + np.int32(1)
# Fewest max-count penalties on one item at which the clamp can take effect — reached soonest when
# every penalty lands before any min-count +1.  Below this, plain add/subtract arithmetic
# reproduces the from-scratch result exactly; from it on, the item must be recomputed by replaying
# its constraints in `_compute_score`'s order (`_recompute_item_score`).
_CLAMP_REACHABLE_MIN_PENALTIES = _SCORE_PENALTY_ALREADY_SAMPLED // _SCORE_PENALTY_HARD_CONSTRAINT


# =================================================================================================
#  _compute_score — from-scratch scoring
# =================================================================================================
@numba.njit("int32[:](int32,int32[:,:],int32[:],int32[:],boolean)", fastmath=True, cache=True)
def _compute_score(
    n: np.int32,
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    already_sampled: NDArray[np.int32],
    hard_max_constraints: bool,
) -> NDArray[np.int32]:
    """Score each integer in `[0, n)` based on how sampling each integer helps toward satisfying the constraints.

      - if it helps achieve a min_count that is not satisfied yet:    +1
      - if it would violate a max_count that we already hit:          -1      if hard_max_constraints=False
                                                                      -2**24  if hard_max_constraints=True
      - if we already sampled it:                                     -2**30  if hard_max_constraints=True.

    The basic idea behind the scoring is that -if at all possible- integers with score <= 0 will not be sampled.

    :param n: range to score [0, n)
    :param con_values: 2D array (m, 2) with min_count and max_count for each constraint
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py
    :param already_sampled: 1D array of integers already sampled (negative values indicate no more samples)
    :param hard_max_constraints: if True, integers that would violate max_count constraints are heavily penalized
    :return: array of scores for each integer
    """
    m = con_values.shape[0]

    # --- init --------------------------------------------
    max_count_penalty = _SCORE_PENALTY_HARD_CONSTRAINT if hard_max_constraints else np.int32(1)
    scores = np.zeros(n, dtype=np.int32)

    # --- min_count / max_count ---------------------------
    for i_con in np.arange(m, dtype=np.int32):
        min_val = _np_con_min_value(con_values, i_con)
        max_val = _np_con_max_value(con_values, i_con)
        indices = _np_con_indices(con_indices, i_con)

        if min_val > 0:
            for idx in indices:
                scores[idx] += 1
        if max_val <= 0:
            for idx in indices:
                scores[idx] = max(
                    scores[idx] - max_count_penalty,
                    _SCORE_CLAMP_FLOOR,  # avoid wrap-around; stays above the already-sampled marker
                )

    # --- already sampled ---------------------------------
    for i in already_sampled:
        if i >= 0:  # negative values indicate end of valid samples
            scores[i] = -_SCORE_PENALTY_ALREADY_SAMPLED

    return scores


# =================================================================================================
#  ConstraintScoreState — incremental upkeep
# =================================================================================================
class ConstraintScoreState(NamedTuple):
    """Mutable score state for one constrained-sampling call, passable into njit functions.

    `scores` always equals the `_compute_score(..., hard_max_constraints=True)` result for the
    current `con_values` and the draws applied so far; once activated (`soft_active`),
    `scores_soft` equally tracks the `hard_max_constraints=False` result.

    Drawn and forbidden items sit at `-_SCORE_PENALTY_ALREADY_SAMPLED`, which no
    constraint-derived score can reach (the clamp floor is one above it), so that value doubles as
    the "no longer drawable" flag.  Mutate only through `apply_draw` / `activate_soft_scores`.
    """

    scores: NDArray[np.int32]  # (n,) maintained _compute_score result, hard-max semantics
    scores_soft: NDArray[np.int32]  # (n,) same with soft-max semantics; meaningless until soft_active
    soft_active: NDArray[np.int32]  # (1,) 1 once scores_soft has been activated (namedtuple-compatible flag)
    penalty_counts: NDArray[np.int32]  # (n_covered,) per item: # of its constraints with exhausted max-counts
    con_values: NDArray[np.int32]  # (m, 2) working min/max counts, decremented as draws accumulate
    con_indices: NDArray[np.int32]  # static: constraint -> member items (never modified)
    item_con_indices: NDArray[np.int32]  # static: item -> ascending constraint ids (never modified)


@numba.njit(cache=True)
def new_constraint_score_state(
    n: np.int32,
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    item_con_indices: NDArray[np.int32],
    i_forbidden: NDArray[np.int32],
) -> ConstraintScoreState:
    """Build the score state for a fresh sampling call: full scoring plus penalty bookkeeping.

    :param n: range to score [0, n)
    :param con_values: 2D array (m, 2) with min_count and max_count for each constraint; copied, the
                       caller's array is never modified
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py
    :param item_con_indices: 1D array with the transposed membership, format described in _constraints.py
    :param i_forbidden: 1D array of integers that must never be drawn; marked as sampled up front
    :return: the initialized ConstraintScoreState
    """
    con_values_working = con_values.copy()
    scores = _compute_score(n, con_values_working, con_indices, i_forbidden, True)
    scores_soft = np.empty(n, dtype=np.int32)  # filled by activate_soft_scores on first use
    soft_active = np.zeros(1, dtype=np.int32)

    # count, per item, the constraints whose max-count is already exhausted (detects items whose
    # clamp can take effect)
    n_covered = np.int32(item_con_indices[0] // 2) if item_con_indices.shape[0] > 0 else np.int32(0)
    penalty_counts = np.zeros(n_covered, dtype=np.int32)
    for i_con in np.arange(con_values_working.shape[0], dtype=np.int32):
        if _np_con_max_value(con_values_working, i_con) <= 0:
            for idx in _np_con_indices(con_indices, i_con):
                penalty_counts[idx] += 1

    return ConstraintScoreState(
        scores, scores_soft, soft_active, penalty_counts, con_values_working, con_indices, item_con_indices
    )


@numba.njit(cache=True)
def _recompute_item_score(state: ConstraintScoreState, idx: np.int32) -> np.int32:
    """Recompute one item's score exactly as `_compute_score` would.

    Only called for items with `_CLAMP_REACHABLE_MIN_PENALTIES` or more exhausted max-counts.  The
    replay visits the item's constraints in ascending order — `_compute_score`'s constraint order —
    with the same per-step clamp, so it lands on the identical value.
    """
    score = np.int32(0)
    for i_con in _np_con_indices(state.item_con_indices, idx):
        i_con32 = np.int32(i_con)
        if _np_con_min_value(state.con_values, i_con32) > 0:
            score += np.int32(1)
        if _np_con_max_value(state.con_values, i_con32) <= 0:
            score = max(score - _SCORE_PENALTY_HARD_CONSTRAINT, _SCORE_CLAMP_FLOOR)
    return score


@numba.njit(inline="always", cache=True)
def _retract_min_bonus(state: ConstraintScoreState, idx: np.int32) -> None:
    """Remove the +1 that a just-satisfied min-count no longer grants to member `idx`."""
    if state.scores[idx] == -_SCORE_PENALTY_ALREADY_SAMPLED:
        return  # drawn/forbidden items stay pinned at the sampled marker
    if state.penalty_counts[idx] >= _CLAMP_REACHABLE_MIN_PENALTIES:
        state.scores[idx] = _recompute_item_score(state, idx)
    else:
        state.scores[idx] -= np.int32(1)
    if state.soft_active[0] == 1:
        state.scores_soft[idx] -= np.int32(1)


@numba.njit(inline="always", cache=True)
def _apply_max_penalty(state: ConstraintScoreState, idx: np.int32) -> None:
    """Apply the max-count penalty of a just-exhausted constraint to member `idx` (hard 2**24, soft 1)."""
    if state.scores[idx] == -_SCORE_PENALTY_ALREADY_SAMPLED:
        return  # drawn/forbidden items stay pinned at the sampled marker
    state.penalty_counts[idx] += 1
    if state.penalty_counts[idx] >= _CLAMP_REACHABLE_MIN_PENALTIES:
        state.scores[idx] = _recompute_item_score(state, idx)
    else:
        state.scores[idx] = max(state.scores[idx] - _SCORE_PENALTY_HARD_CONSTRAINT, _SCORE_CLAMP_FLOOR)
    if state.soft_active[0] == 1:
        state.scores_soft[idx] -= np.int32(1)


@numba.njit(cache=True)
def apply_draw(state: ConstraintScoreState, s: np.int32) -> None:
    """Apply a draw of item `s` to the state, updating all arrays in place.

    - decrements the working min/max counts of `s`'s constraints
    - sweeps a constraint's member list only when a count crosses its satisfaction threshold
      (min-count just reached: its +1 is retracted from the members; max-count just exhausted:
      its penalty is applied to them)
    - pins `s` itself at `-_SCORE_PENALTY_ALREADY_SAMPLED`, so it cannot be drawn again
    """
    # constraints containing s (items beyond the transposed membership's coverage belong to no constraint)
    n_covered = state.penalty_counts.shape[0]
    s_cons = _np_con_indices(state.item_con_indices, s) if s < n_covered else state.item_con_indices[0:0]

    # 1) decrement the working counts of all of s's constraints, so the threshold sweeps below
    #    (and any rescoring they trigger) read the fully updated counts
    for i_con in s_cons:
        state.con_values[i_con, 0] -= 1
        state.con_values[i_con, 1] -= 1

    # 2) sweep the members of each constraint that a decrement pushed across a threshold
    for i_con in s_cons:
        i_con32 = np.int32(i_con)
        if _np_con_min_value(state.con_values, i_con32) == 0:  # min-count just reached (was 1)
            for idx in _np_con_indices(state.con_indices, i_con32):
                _retract_min_bonus(state, np.int32(idx))
        if _np_con_max_value(state.con_values, i_con32) == 0:  # max-count just exhausted (was 1)
            for idx in _np_con_indices(state.con_indices, i_con32):
                _apply_max_penalty(state, np.int32(idx))

    # 3) s itself can no longer be drawn
    state.scores[s] = -_SCORE_PENALTY_ALREADY_SAMPLED
    if state.soft_active[0] == 1:
        state.scores_soft[s] = -_SCORE_PENALTY_ALREADY_SAMPLED


@numba.njit(cache=True)
def activate_soft_scores(state: ConstraintScoreState, n: np.int32, already_sampled: NDArray[np.int32]) -> None:
    """Fill `scores_soft` with a fresh soft scoring and start maintaining it in `apply_draw`.

    Called on the first draw whose remaining min-counts exceed what the remaining draws can
    deliver; from then on the soft scores stay current incrementally.  Unlike the hard variant,
    the soft penalty of 1 can never reach the clamp floor (that would take
    `_SCORE_PENALTY_ALREADY_SAMPLED` exhausted max-counts on one item), so plain add/subtract
    reproduces the from-scratch result exactly and no per-item replay is needed.
    """
    state.scores_soft[:] = _compute_score(n, state.con_values, state.con_indices, already_sampled, False)
    state.soft_active[0] = 1
