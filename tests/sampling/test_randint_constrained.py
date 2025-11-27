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
def test_randint_constrained_basic(seed: int, mode: str) -> None:
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
    if mode == "robust":
        samples = randint_constrained_robust(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=np.zeros(0, dtype=np.float32),
            seed=np.int64(seed),
        )
    else:
        samples = randint_constrained(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=np.zeros(0, dtype=np.float32),
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
def test_randint_constrained_infeasible(seed: int, mode: str) -> None:
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
    if mode == "robust":
        samples = randint_constrained_robust(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=np.zeros(0, dtype=np.float32),
            seed=np.int64(seed),
        )
    else:
        samples = randint_constrained(
            n=np.int32(n),
            k=np.int32(k),
            con_values=con_values,
            con_indices=con_indices,
            p=np.zeros(0, dtype=np.float32),
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
