import numpy as np
import pytest

from max_div.constraints import Constraint
from max_div.constraints._numba import _build_array_repr
from max_div.sampling.con import randint_constrained


# =================================================================================================
#  randint_constrained
# =================================================================================================
@pytest.mark.parametrize("seed", list(range(1, 50)))
@pytest.mark.parametrize("eager", [True, False])
def test_randint_constrained_basic(seed: int, eager: bool) -> None:
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

    # --- act ---------------------------------------------
    samples = randint_constrained(
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=np.zeros(0, dtype=np.float32),
        seed=np.int64(seed),
        eager=eager,
    )

    # --- assert ------------------------------------------
    assert len(samples) == k
    assert len(set(samples)) == k  # unique samples
    assert all(0 <= s < n for s in samples)

    for con in cons:
        count = sum(1 for s in samples if s in con.int_set)
        assert con.min_count <= count <= con.max_count


@pytest.mark.parametrize("seed", list(range(1, 50)))
@pytest.mark.parametrize("eager", [True, False])
def test_randint_constrained_infeasible(seed: int, eager: bool) -> None:
    # --- arrange -----------------------------------------
    n = 100
    k = 5
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=3, max_count=3),
        Constraint(int_set={10, 11, 12, 13, 14}, min_count=3, max_count=3),
    ]

    # convert to numba format
    con_values, con_indices = _build_array_repr(cons)

    # --- act ---------------------------------------------
    samples = randint_constrained(
        n=np.int32(n),
        k=np.int32(k),
        con_values=con_values,
        con_indices=con_indices,
        p=np.zeros(0, dtype=np.float32),
        seed=np.int64(seed),
        eager=eager,
    )

    # --- assert ------------------------------------------
    assert len(samples) == k
    assert len(set(samples)) == k  # unique samples
    assert all(0 <= s < n for s in samples)

    for con in cons:
        count = sum(1 for s in samples if s in con.int_set)
        assert 2 <= count <= 3  # least harmful solution is to sample 2 or 3 from each constraint set
