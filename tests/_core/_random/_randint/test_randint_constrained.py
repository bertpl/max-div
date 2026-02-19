from functools import partial

import numpy as np
import pytest

from max_div._core._random import new_rng_state
from max_div._core._random._randint._randint_constrained import (
    _SCORE_PENALTY_ALREADY_SAMPLED,
    _compute_score,
    randint_constrained,
)
from max_div._core.constraints import Constraint, ConstraintList


# =================================================================================================
#  randint_constrained
# =================================================================================================
@pytest.mark.parametrize("seed", list(range(1, 50)))
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("p_mode", ["random", "empty", "zero"])
def test_randint_constrained_basic(seed: int, eager: bool, p_mode: str) -> None:
    # --- arrange -----------------------------------------
    n = 20
    k = 5
    constraints = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13, 14}, min_count=1, max_count=3),
        Constraint(int_set={3, 4, 10, 11}, min_count=2, max_count=2),
    ]
    rng_state = new_rng_state(np.int64(seed))

    # convert to numpy format
    con_values, con_indices = ConstraintList(constraints).to_numpy()

    # construct p array
    if p_mode == "random":
        p = np.random.rand(n).astype(np.float32)
    elif p_mode == "empty":
        p = np.zeros(0, dtype=np.float32)
    else:  # p_mode == "zero"
        p = np.zeros(n, dtype=np.float32)

    # copies for later comparison
    con_values_before = con_values.copy()
    con_indices_before = con_indices.copy()
    p_before = p.copy()

    # --- act ---------------------------------------------
    samples = randint_constrained(
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        rng_state=rng_state,
        eager=eager,
    )

    # --- assert ------------------------------------------
    assert len(samples) == k
    assert len(set(samples)) == k  # unique samples
    assert all(0 <= s < n for s in samples)

    # check if provided arrays were left untouched
    assert np.array_equal(con_values_before, con_values), "con_values array should never be modified."
    assert np.array_equal(con_indices, con_indices_before), "p array should never be modified."
    assert np.array_equal(p, p_before), "p array should never be modified."

    for con in constraints:
        count = sum(1 for s in samples if s in con.int_set)
        assert con.min_count <= count <= con.max_count


@pytest.mark.parametrize("seed", list(range(1, 50)))
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("p_mode", ["random", "empty", "zero"])
def test_randint_constrained_infeasible(seed: int, eager: bool, p_mode: str) -> None:
    # --- arrange -----------------------------------------
    n = 100
    k = 5
    constraints = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=3, max_count=3),
        Constraint(int_set={10, 11, 12, 13, 14}, min_count=3, max_count=3),
    ]
    rng_state = new_rng_state(np.int64(seed))

    # convert to numpy format
    con_values, con_indices = ConstraintList(constraints).to_numpy()

    # construct p array
    if p_mode == "random":
        p = np.random.rand(n).astype(np.float32)
    elif p_mode == "empty":
        p = np.zeros(0, dtype=np.float32)
    else:  # p_mode == "zero"
        p = np.zeros(n, dtype=np.float32)

    # copies for later comparison
    con_values_before = con_values.copy()
    con_indices_before = con_indices.copy()
    p_before = p.copy()

    # --- act ---------------------------------------------
    samples = randint_constrained(
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        rng_state=rng_state,
        eager=eager,
    )

    # --- assert ------------------------------------------
    assert len(samples) == k
    assert len(set(samples)) == k  # unique samples
    assert all(0 <= s < n for s in samples)

    # check if provided arrays were left untouched
    assert np.array_equal(con_values_before, con_values), "con_values array should never be modified."
    assert np.array_equal(con_indices, con_indices_before), "p array should never be modified."
    assert np.array_equal(p, p_before), "p array should never be modified."

    for con in constraints:
        count = sum(1 for s in samples if s in con.int_set)
        assert 2 <= count <= 3  # least harmful solution is to sample 2 or 3 from each constraint set


@pytest.mark.parametrize("k_context", [-1, 0, 1, 2, 3, 5, 10])
@pytest.mark.parametrize("seed", list(range(42, 50)))
def test_randint_constrained_k_context(k_context: int, seed: int):
    """
    Test if k_context parameter is working as expected.

    Therefore, we set up a constraint that can only be satisfied if k_context>k, but with very small
     member probabilities.

        --> k_context = k     => sample will generate a member from the constraint, despite low probability
        --> k_context > k     => sample will avoid the constraint, as it's not urgent + other samples have higher prob.

    """

    # --- arrange -----------------------------------------
    n = 6
    k = 1
    constraints = [
        Constraint(int_set={0, 1, 2}, min_count=1, max_count=3),
        Constraint(int_set={0, 1, 2, 3, 4, 5}, min_count=1, max_count=3),
    ]
    p = np.array([1e-15, 1e-15, 1e-15, 1.0, 1.0, 1.0], dtype=np.float32)
    rng_state = new_rng_state(np.int64(seed))

    # convert to numpy format
    con_values, con_indices = ConstraintList(constraints).to_numpy()

    # copies for later comparison
    con_values_before = con_values.copy()
    con_indices_before = con_indices.copy()
    p_before = p.copy()

    # --- act ---------------------------------------------
    samples = randint_constrained(
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        rng_state=rng_state,
        eager=False,
        k_context=np.int32(k_context),
    )

    # --- assert ------------------------------------------

    # check k_context correctness
    if k_context < 2:
        # must sample from constraint 1 (0,1,2)
        assert samples[0] in {0, 1, 2}
    else:
        # we can sample from either constraint 1 or 2; p will motivate algorithm to choose from (3,4,5)
        assert samples[0] in {3, 4, 5}

    # check if provided arrays were left untouched
    assert np.array_equal(con_values_before, con_values), "con_values array should never be modified."
    assert np.array_equal(con_indices, con_indices_before), "p array should never be modified."
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "k, n, n_forbidden, expected_ok",
    [
        (9, 10, 0, True),
        (10, 10, 0, True),
        (8, 10, 1, True),
        (8, 10, 2, True),
        (2, 10, 8, True),
        (10, 9, 0, False),
        (8, 10, 3, False),
    ],
)
def test_randint_constrained_i_forbidden_validation(k, n, n_forbidden, expected_ok: bool):
    """Check for ValueError in case k, n, len(i_forbidden) are conflicting."""

    # --- arrange -----------------------------------------
    constraints = [Constraint(int_set={0, 1, 2}, min_count=2, max_count=3)]

    # convert to numpy format
    con_values, con_indices = ConstraintList(constraints).to_numpy()

    p = np.random.rand(n).astype(np.float32)
    rng_state = new_rng_state(42)
    i_forbidden = np.array(list(range(n_forbidden)), dtype=np.int32)

    function_call = partial(
        randint_constrained,
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        rng_state=rng_state,
        eager=False,
        i_forbidden=i_forbidden,
    )

    # --- act & assert ------------------------------------
    if expected_ok:
        _ = function_call()  # should work
    else:
        with pytest.raises(ValueError):
            _ = function_call()  # should raise ValueError


@pytest.mark.parametrize("min_count", [0, 1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("eager", [False, True])
def test_randint_constrained_i_forbidden_priorities(min_count: int, eager: bool):
    """Test if i_forbidden is prioritized over constraints and p=0."""

    # --- arrange -----------------------------------------
    rng_state = new_rng_state(42)
    n = 10
    k = 5
    i_forbidden = np.array([3, 4], dtype=np.int32)

    # set up constraints & p such that sampling is tempted sample forbidden indices 3 or 4
    constraints = [Constraint(int_set={0, 1, 2, 3, 4}, min_count=min_count, max_count=10)]
    con_values, con_indices = ConstraintList(constraints).to_numpy()

    p = np.array([0, 0.1, 0.1, 1, 1, 0.1, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

    # copies for later comparison
    i_forbidden_before = i_forbidden.copy()
    con_values_before = con_values.copy()
    con_indices_before = con_indices.copy()
    p_before = p.copy()

    # --- act ---------------------------------------------
    samples = randint_constrained(
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        rng_state=rng_state,
        eager=eager,
        i_forbidden=i_forbidden,
    )

    # --- assert ------------------------------------------

    # never sample forbidden indices
    assert not any(s == 3 for s in samples)
    assert not any(s == 4 for s in samples)

    # if k==2, there's only 1 possible solution that also avoids sampling from p=0
    if k == 2:
        assert set(samples) == {1, 2}

    # check if provided arrays were left untouched
    assert np.array_equal(i_forbidden_before, i_forbidden), "i_forbidden array should never be modified."
    assert np.array_equal(con_values_before, con_values), "con_values array should never be modified."
    assert np.array_equal(con_indices, con_indices_before), "p array should never be modified."
    assert np.array_equal(p, p_before), "p array should never be modified."


def test_randint_constrained_score_wrap_around():
    """
    In case of very large nr of constraints, very negative scores can theoretically wrap around to positive ones.
    This could e.g. cause duplicate samples to be generated.

    This test specifically checks if the safeguard against this issue is working as expected.
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
    con_values, con_indices = ConstraintList(constraints).to_numpy()

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
