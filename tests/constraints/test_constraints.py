import numpy as np
import pytest

from max_div.constraints import Constraints


# =================================================================================================
#  Constraints
# =================================================================================================
@pytest.mark.parametrize("n_cons", [0, 1, 2, 4, 8])
def test_constraints_n_cons(n_cons: int):
    # --- arrange -----------------------------------------
    cons = Constraints()
    for _ in range(n_cons):
        cons.add(indices={0, 1, 2}, min_count=1, max_count=2)

    # --- act ---------------------------------------------
    result = cons.n_cons

    # --- assert ------------------------------------------
    assert result == n_cons


@pytest.mark.parametrize("deepcopy", [False, True])
def test_constraints_all(deepcopy: bool):
    # --- arrange -----------------------------------------
    cons = Constraints()
    cons.add(indices={0, 1, 2}, min_count=1, max_count=2)
    cons.add(indices={3, 4, 5}, min_count=0, max_count=1)

    # --- act ---------------------------------------------
    result = cons.all(deepcopy=deepcopy)
    result_2 = cons.all(deepcopy=deepcopy)

    # --- assert ------------------------------------------
    assert len(result) == 2
    assert result[0].int_set == {0, 1, 2}
    assert result[0].min_count == 1
    assert result[0].max_count == 2
    assert result[1].int_set == {3, 4, 5}
    assert result[1].min_count == 0
    assert result[1].max_count == 1

    if deepcopy:
        assert result is not result_2
        assert result[0] is not result_2[0]
        assert result[0].int_set is not result_2[0].int_set
    else:
        assert result is result_2
        assert result[0] is result_2[0]
        assert result[0].int_set is result_2[0].int_set


def test_constraints_to_numpy():
    # --- arrange -----------------------------------------
    cons = Constraints()
    cons.add(indices={0, 1, 2, 3, 4}, min_count=2, max_count=3)
    cons.add(indices={10, 11, 12, 13}, min_count=0, max_count=7)
    cons.add(indices={3, 11}, min_count=2, max_count=2)

    # --- act ---------------------------------------------
    con_values, con_indices = cons.to_numpy()

    # --- assert ------------------------------------------
    assert np.array_equal(
        con_values,
        np.array(
            [
                [2, 3],  # min_count, max_count for constraint 0
                [0, 7],  # min_count, max_count for constraint 1
                [2, 2],  # min_count, max_count for constraint 2
            ],
            dtype=np.int32,
        ),
    )

    assert con_indices.shape[0] == 17  # (2*n_cons) + (5+4+2) = 6 + 11 = 17
    assert con_indices.dtype == np.int32

    for i, con in enumerate(cons.all()):
        i_start = con_indices[2 * i]
        i_end = con_indices[2 * i + 1]
        assert list(con_indices[i_start:i_end]) == sorted(con.int_set)
