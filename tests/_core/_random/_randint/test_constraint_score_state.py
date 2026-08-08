import numpy as np
import pytest

from max_div._core._random._randint._constraint_score_state import (
    _CLAMP_REACHABLE_MIN_PENALTIES,
    _SCORE_PENALTY_ALREADY_SAMPLED,
    _compute_score,
    activate_soft_scores,
    apply_draw,
    new_constraint_score_state,
)
from max_div._core.constraints import Constraint, ConstraintList

_NO_FORBIDDEN = np.empty(0, dtype=np.int32)


def _assert_scores_match_oracle(state, n: int, con_indices, drawn: list[int], i_forbidden) -> None:
    """Assert the maintained scores (hard, and soft when active) equal a fresh `_compute_score`."""
    already_sampled = np.concatenate((np.array(drawn, dtype=np.int32), i_forbidden))
    oracle = _compute_score(
        n=np.int32(n),
        con_values=state.con_values,
        con_indices=con_indices,
        already_sampled=already_sampled,
        hard_max_constraints=True,
    )
    assert np.array_equal(state.scores, oracle)
    if state.soft_active[0] == 1:
        oracle_soft = _compute_score(
            n=np.int32(n),
            con_values=state.con_values,
            con_indices=con_indices,
            already_sampled=already_sampled,
            hard_max_constraints=False,
        )
        assert np.array_equal(state.scores_soft, oracle_soft)


# =================================================================================================
#  _compute_score
# =================================================================================================
def test_compute_score_wrap_around():
    """Pin the wrap-around safeguard: massed max-count penalties never wrap to positive scores.

    A wrapped-around (positive) score could cause duplicate samples to be generated.
    """
    # --- arrange -----------------------------------------
    n = 10
    constraints = [
        Constraint(
            int_set=set(range(1, n)),  # all samples except index 0
            min_count=0,
            max_count=0,
        )
        for _ in range(1000)
    ]
    con_values, con_indices, _item_con_indices = ConstraintList(constraints).to_numpy()

    # --- act ---------------------------------------------
    score = _compute_score(
        n=np.int32(n),
        con_values=con_values,
        con_indices=con_indices,
        already_sampled=np.array([0], dtype=np.int32),
        hard_max_constraints=True,
    )

    # --- assert ------------------------------------------
    assert max(score) <= 0.0, "Scores should not have wrapped around to positive values."
    assert score[0] == -_SCORE_PENALTY_ALREADY_SAMPLED
    for i in range(1, n):
        assert -_SCORE_PENALTY_ALREADY_SAMPLED < score[i] < 0.0


# =================================================================================================
#  ConstraintScoreState
# =================================================================================================
def test_new_constraint_score_state_scores_and_counts():
    # --- arrange -----------------------------------------
    n = 8
    constraints = [
        Constraint(int_set={0, 1, 2}, min_count=2, max_count=3),  # min still unmet: members get +1
        Constraint(int_set={2, 3}, min_count=0, max_count=0),  # max already exhausted: members penalized
    ]
    con_values, con_indices, item_con_indices = ConstraintList(constraints).to_numpy()
    i_forbidden = np.array([5], dtype=np.int32)

    # --- act ---------------------------------------------
    state = new_constraint_score_state(np.int32(n), con_values, con_indices, item_con_indices, i_forbidden)

    # --- assert ------------------------------------------
    _assert_scores_match_oracle(state, n, con_indices, [], i_forbidden)
    assert np.array_equal(state.penalty_counts, np.array([0, 0, 1, 1], dtype=np.int32))
    assert state.con_values is not con_values, "working counts must be a copy"
    assert np.array_equal(state.con_values, con_values)


def test_apply_draw_updates_working_counts():
    # --- arrange -----------------------------------------
    n = 8
    constraints = [
        Constraint(int_set={0, 1, 2}, min_count=2, max_count=3),
        Constraint(int_set={2, 3}, min_count=1, max_count=2),
    ]
    con_values, con_indices, item_con_indices = ConstraintList(constraints).to_numpy()
    state = new_constraint_score_state(np.int32(n), con_values, con_indices, item_con_indices, _NO_FORBIDDEN)

    # --- act ---------------------------------------------
    apply_draw(state, np.int32(2))  # member of both constraints

    # --- assert ------------------------------------------
    assert np.array_equal(state.con_values, np.array([[1, 2], [0, 1]], dtype=np.int32))
    assert state.scores[2] == -_SCORE_PENALTY_ALREADY_SAMPLED
    assert np.array_equal(con_values, np.array([[2, 3], [1, 2]], dtype=np.int32)), "caller's array modified"


def test_apply_draw_beyond_covered_items():
    """Drawing an item no constraint references only pins it at the sampled marker."""
    # --- arrange -----------------------------------------
    n = 10
    constraints = [Constraint(int_set={0, 1}, min_count=1, max_count=2)]
    con_values, con_indices, item_con_indices = ConstraintList(constraints).to_numpy()
    state = new_constraint_score_state(np.int32(n), con_values, con_indices, item_con_indices, _NO_FORBIDDEN)
    scores_before = state.scores.copy()

    # --- act ---------------------------------------------
    apply_draw(state, np.int32(7))

    # --- assert ------------------------------------------
    assert state.scores[7] == -_SCORE_PENALTY_ALREADY_SAMPLED
    assert np.array_equal(state.con_values, con_values)
    scores_before[7] = -_SCORE_PENALTY_ALREADY_SAMPLED
    assert np.array_equal(state.scores, scores_before)


def test_apply_draw_without_constraints():
    # --- arrange -----------------------------------------
    n = 5
    con_values, con_indices, item_con_indices = ConstraintList([]).to_numpy()
    state = new_constraint_score_state(np.int32(n), con_values, con_indices, item_con_indices, _NO_FORBIDDEN)

    # --- act ---------------------------------------------
    apply_draw(state, np.int32(3))

    # --- assert ------------------------------------------
    _assert_scores_match_oracle(state, n, con_indices, [3], _NO_FORBIDDEN)


@pytest.mark.parametrize("seed", list(range(20)))
def test_apply_draw_matches_oracle_on_random_problems(seed: int):
    """Property pin: after ANY draw sequence, the maintained scores equal a full `_compute_score`.

    The draw sequence deliberately ignores the sampling policy — the invariant must hold for every
    sequence of distinct items, not only ones the sampler would produce.
    """
    # --- arrange -----------------------------------------
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 40))
    m = int(rng.integers(1, 10))
    constraints = []
    for _ in range(m):
        size = int(rng.integers(1, n))
        int_set = set(rng.choice(n, size=size, replace=False).tolist())
        min_count = int(rng.integers(0, size + 2))
        max_count = int(rng.integers(0, size + 2))  # may exhaust quickly, or start exhausted
        constraints.append(Constraint(int_set=int_set, min_count=min_count, max_count=max_count))
    con_values, con_indices, item_con_indices = ConstraintList(constraints).to_numpy()

    n_forbidden = int(rng.integers(0, n // 3 + 1))
    i_forbidden = rng.choice(n, size=n_forbidden, replace=False).astype(np.int32)

    state = new_constraint_score_state(np.int32(n), con_values, con_indices, item_con_indices, i_forbidden)

    drawable = [i for i in range(n) if i not in set(i_forbidden.tolist())]
    n_draws = int(rng.integers(1, len(drawable) + 1))
    draws = rng.choice(drawable, size=n_draws, replace=False).tolist()
    soft_activation_point = int(rng.integers(0, n_draws))  # soft scores join mid-sequence, as in the sampler

    # --- act & assert ------------------------------------
    drawn: list[int] = []
    for i_draw, s in enumerate(draws):
        if i_draw == soft_activation_point:
            already_sampled = np.concatenate((np.array(drawn, dtype=np.int32), i_forbidden))
            activate_soft_scores(state, np.int32(n), already_sampled)
        apply_draw(state, np.int32(s))
        drawn.append(int(s))
        _assert_scores_match_oracle(state, n, con_indices, drawn, i_forbidden)


@pytest.mark.parametrize(
    "n_exhaustible", [int(_CLAMP_REACHABLE_MIN_PENALTIES), int(_CLAMP_REACHABLE_MIN_PENALTIES) + 6]
)
def test_apply_draw_matches_oracle_with_many_exhausted_max_counts(n_exhaustible: int):
    """Items whose exhausted max-counts reach `_CLAMP_REACHABLE_MIN_PENALTIES` are recomputed exactly.

    One draw exhausts every max-1 constraint at once (penalty path), and a later draw satisfies the
    min-count constraint (retraction path) — both take the per-item replay path, where the per-step
    clamp makes plain add/subtract wrong.
    """
    # --- arrange -----------------------------------------
    n = 12
    constraints = [Constraint(int_set=set(range(n)), min_count=0, max_count=1) for _ in range(n_exhaustible)]
    constraints.append(Constraint(int_set={1, 2, 3}, min_count=1, max_count=3))
    con_values, con_indices, item_con_indices = ConstraintList(constraints).to_numpy()

    state = new_constraint_score_state(np.int32(n), con_values, con_indices, item_con_indices, _NO_FORBIDDEN)

    # --- act & assert ------------------------------------
    activate_soft_scores(state, np.int32(n), _NO_FORBIDDEN)

    apply_draw(state, np.int32(0))  # exhausts all max-1 constraints: every other item takes them as penalties
    _assert_scores_match_oracle(state, n, con_indices, [0], _NO_FORBIDDEN)
    assert np.all(state.penalty_counts[1:] >= _CLAMP_REACHABLE_MIN_PENALTIES)

    apply_draw(state, np.int32(2))  # satisfies the min-count: retraction sweeps items whose clamp can take effect
    _assert_scores_match_oracle(state, n, con_indices, [0, 2], _NO_FORBIDDEN)
