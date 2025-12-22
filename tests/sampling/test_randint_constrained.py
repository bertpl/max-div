from functools import partial

import numpy as np
import pytest

from max_div.sampling._constraint_helpers import _build_array_repr
from max_div.sampling.con import randint_constrained, randint_constrained_robust
from max_div.solver import Constraint


# =================================================================================================
#  randint_constrained
# =================================================================================================
@pytest.mark.parametrize("seed", list(range(1, 50)))
@pytest.mark.parametrize("mode", ["non_eager", "eager", "robust"])
@pytest.mark.parametrize("p_mode", ["random", "empty", "zero"])
def test_randint_constrained_basic(seed: int, mode: str, p_mode: str) -> None:
    # --- arrange -----------------------------------------
    n = 20
    k = 5
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13, 14}, min_count=1, max_count=3),
        Constraint(int_set={3, 4, 10, 11}, min_count=2, max_count=2),
    ]

    # convert to numba format
    con_values, con_indices = _build_array_repr(cons)

    # construct p array
    if p_mode == "random":
        p = np.random.rand(n).astype(np.float32)
    elif p_mode == "empty":
        p = np.zeros(0, dtype=np.float32)
    else:  # p_mode == "zero"
        p = np.zeros(n, dtype=np.float32)

    # --- act ---------------------------------------------
    if mode == "robust":
        samples = randint_constrained_robust(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=p,
            seed=np.int64(seed),
        )
    else:
        samples = randint_constrained(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=p,
            seed=np.int64(seed),
            eager=(mode == "eager"),
        )

    # --- assert ------------------------------------------
    assert len(samples) == k
    assert len(set(samples)) == k  # unique samples
    assert all(0 <= s < n for s in samples)

    for con in cons:
        count = sum(1 for s in samples if s in con.int_set)
        assert con.min_count <= count <= con.max_count


@pytest.mark.parametrize("seed", list(range(1, 50)))
@pytest.mark.parametrize("mode", ["non_eager", "eager", "robust"])
@pytest.mark.parametrize("p_mode", ["random", "empty", "zero"])
def test_randint_constrained_infeasible(seed: int, mode: str, p_mode: str) -> None:
    # --- arrange -----------------------------------------
    n = 100
    k = 5
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=3, max_count=3),
        Constraint(int_set={10, 11, 12, 13, 14}, min_count=3, max_count=3),
    ]

    # convert to numba format
    con_values, con_indices = _build_array_repr(cons)

    # construct p array
    if p_mode == "random":
        p = np.random.rand(n).astype(np.float32)
    elif p_mode == "empty":
        p = np.zeros(0, dtype=np.float32)
    else:  # p_mode == "zero"
        p = np.zeros(n, dtype=np.float32)

    # --- act ---------------------------------------------
    if mode == "robust":
        samples = randint_constrained_robust(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=p,
            seed=np.int64(seed),
        )
    else:
        samples = randint_constrained(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=p,
            seed=np.int64(seed),
            eager=(mode == "eager"),
        )

    # --- assert ------------------------------------------
    assert len(samples) == k
    assert len(set(samples)) == k  # unique samples
    assert all(0 <= s < n for s in samples)

    for con in cons:
        count = sum(1 for s in samples if s in con.int_set)
        assert 2 <= count <= 3  # least harmful solution is to sample 2 or 3 from each constraint set


def test_randint_constrained_robust_validation():
    """Check for ValueError in case n_trials < 3."""

    # --- arrange -----------------------------------------
    n = 10
    k = 5
    cons = [Constraint(int_set={0, 1, 2}, min_count=2, max_count=3)]

    # convert to numba format
    con_values, con_indices = _build_array_repr(cons)

    p = np.random.rand(n).astype(np.float32)
    seed = 42

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        randint_constrained_robust(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=p,
            seed=np.int64(seed),
            n_trials=2,  # <3 --> ValueError
        )


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
    cons = [
        Constraint(int_set={0, 1, 2}, min_count=1, max_count=3),
        Constraint(int_set={0, 1, 2, 3, 4, 5}, min_count=1, max_count=3),
    ]
    p = np.array([1e-15, 1e-15, 1e-15, 1.0, 1.0, 1.0], dtype=np.float32)

    # convert to numba format
    con_values, con_indices = _build_array_repr(cons)

    # --- act ---------------------------------------------
    samples = randint_constrained(
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        seed=np.int64(seed),
        eager=False,
        k_context=np.int32(k_context),
    )

    # --- assert ------------------------------------------
    if k_context < 2:
        # must sample from constraint 1 (0,1,2)
        assert samples[0] in {0, 1, 2}
    else:
        # we can sample from either constraint 1 or 2; p will motivate algorithm to choose from (3,4,5)
        assert samples[0] in {3, 4, 5}


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
    cons = [Constraint(int_set={0, 1, 2}, min_count=2, max_count=3)]

    # convert to numba format
    con_values, con_indices = _build_array_repr(cons)

    p = np.random.rand(n).astype(np.float32)
    seed = 42
    i_forbidden = np.array(list(range(n_forbidden)), dtype=np.int32)

    function_call = partial(
        randint_constrained,
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        seed=np.int64(seed),
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
    seed = 42
    n = 10
    k = 5
    i_forbidden = np.array([3, 4], dtype=np.int32)

    # set up constraints & p such that sampling is tempted sample forbidden indices 3 or 4
    cons = [Constraint(int_set={0, 1, 2, 3, 4}, min_count=min_count, max_count=10)]
    con_values, con_indices = _build_array_repr(cons)

    p = np.array([0, 0.1, 0.1, 1, 1, 0.1, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

    # --- act ---------------------------------------------
    samples = randint_constrained(
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        seed=np.int64(seed),
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
