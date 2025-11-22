import pytest

from max_div.solver import Constraints


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
