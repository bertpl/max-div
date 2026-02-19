import numpy as np
import pytest

from max_div._core._random import P_UNIFORM, choice_constrained, new_rng_state
from max_div._core._utils import deterministic_hash_int64
from max_div._core.constraints import Constraint, ConstraintList


def test_choice_constrained_argument_validation():
    # --- arrange -----------------------------------------
    k = 5
    n = 100
    constraints = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13, 14}, min_count=1, max_count=3),
        Constraint(int_set={3, 4, 10, 11}, min_count=2, max_count=2),
    ]
    con_values, con_indices = ConstraintList(constraints).to_numpy()
    rng_state = new_rng_state(np.int64(42))
    values = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14], dtype=np.int32)
    p = P_UNIFORM

    # wrong values
    k_too_large = 20
    p_wrong_size = np.array([0.1, 0.9], dtype=np.float32)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        choice_constrained(
            n=np.int32(n),
            values=values,
            k=np.int32(k_too_large),
            p=p,
            rng_state=rng_state,
            con_values=con_values,
            con_indices=con_indices,
        )

    with pytest.raises(ValueError):
        choice_constrained(
            n=np.int32(n),
            values=values,
            k=np.int32(k),
            p=p_wrong_size,
            rng_state=rng_state,
            con_values=con_values,
            con_indices=con_indices,
        )


@pytest.mark.parametrize("k_context", [-1, 5, 8, 10])
@pytest.mark.parametrize("n", [15, 20, 50, 100, 200, 500, 1000])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("uniform", [False, True])
def test_choice_constrained_invariants(eager: bool, n: int, k_context: int, uniform: bool):
    # --- arrange -----------------------------------------
    k = 5
    constraints = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13, 14}, min_count=1, max_count=3),
        Constraint(int_set={3, 4, 10, 11}, min_count=2, max_count=2),
    ]
    con_values, con_indices = ConstraintList(constraints).to_numpy()
    seed = deterministic_hash_int64((k_context, n, eager, uniform))
    rng_state = new_rng_state(np.int64(seed))

    values = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14] + [round(i) for i in np.linspace(14, n - 1, 10)]
    values = np.array(sorted(set(values)), dtype=np.int32)

    if uniform:
        p = P_UNIFORM
    else:
        p = np.array(np.random.random(values.size), dtype=np.float32)  # random probabilities

    # copies to check for modification later
    p_before = p.copy()
    rng_state_before = rng_state.copy()

    # --- act ---------------------------------------------
    samples = choice_constrained(
        n=np.int32(n),
        values=values,
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        rng_state=rng_state,
        eager=eager,
        k_context=np.int32(k_context),
    )

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int32

    assert all([s in values for s in samples])
    assert len(samples) == len(set(samples))

    for con in constraints:
        count = sum(1 for s in samples if s in con.int_set)
        assert con.min_count <= count <= con.max_count

    assert np.array_equal(p, p_before), "p array was modified"
    assert not np.array_equal(rng_state, rng_state_before), "rng_state was not updated"


@pytest.mark.parametrize("seed", list(range(1, 100)))
def test_choice_constrained_p_adherence(seed: int):
    # --- arrange -----------------------------------------
    n = 20
    k = 5
    constraints = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13, 14}, min_count=1, max_count=3),
        Constraint(int_set={3, 4, 10, 11}, min_count=2, max_count=2),
    ]
    con_values, con_indices = ConstraintList(constraints).to_numpy()
    rng_state = new_rng_state(np.int64(seed))
    values = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14], dtype=np.int32)
    p = np.array([1, 1, 1, 1, 1, 1, 1, 1e3, 1, 1], dtype=np.float32)  # very strongly favor value 12

    # --- act ---------------------------------------------
    samples = choice_constrained(
        n=np.int32(n),
        values=values,
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=p,
        rng_state=rng_state,
    )

    # --- assert ------------------------------------------
    assert np.int32(12) in samples
