import numpy as np
import pytest

from max_div.random._constraints import (
    Constraint,
    ConstraintList,
    _build_array_repr,
    _np_con_indices,
    _np_con_max_value,
    _np_con_min_value,
    _np_con_total_violation,
    _np_largest_con_index,
)


def test_build_array_repr():
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
        Constraint(int_set={3, 11}, min_count=2, max_count=2),
    ]

    # --- act ---------------------------------------------
    con_values_1, con_indices_1 = _build_array_repr(cons)
    con_values_2, con_indices_2 = ConstraintList(cons).to_numpy()

    # --- assert ------------------------------------------
    assert np.array_equal(
        con_values_1,
        np.array(
            [
                [2, 3],  # min_count, max_count for constraint 0
                [0, 7],  # min_count, max_count for constraint 1
                [2, 2],  # min_count, max_count for constraint 2
            ],
            dtype=np.int32,
        ),
    )

    assert con_indices_1.shape[0] == 17  # (2*m) + (5+4+2) = 6 + 11 = 17
    assert con_indices_1.dtype == np.int32

    for i, con in enumerate(cons):
        i_start = con_indices_1[2 * i]
        i_end = con_indices_1[2 * i + 1]
        assert list(con_indices_1[i_start:i_end]) == sorted(con.int_set)

    assert np.array_equal(con_values_1, con_values_2)
    assert np.array_equal(con_indices_1, con_indices_2)


def test_np_con_min_value():
    # --- arrange -----------------------------------------
    con_values, con_indices = ConstraintList(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    ).to_numpy()

    # --- act & assert ------------------------------------
    assert _np_con_min_value(con_values, np.int32(0)) == 2
    assert _np_con_min_value(con_values, np.int32(1)) == 0
    assert _np_con_min_value(con_values, np.int32(2)) == 2


def test_np_con_max_value():
    # --- arrange -----------------------------------------
    con_values, con_indices = ConstraintList(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    ).to_numpy()

    # --- act & assert ------------------------------------
    assert _np_con_max_value(con_values, np.int32(0)) == 3
    assert _np_con_max_value(con_values, np.int32(1)) == 7
    assert _np_con_max_value(con_values, np.int32(2)) == 2


def test_np_con_indices():
    # --- arrange -----------------------------------------
    con_values, con_indices = ConstraintList(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    ).to_numpy()

    # --- act & assert ------------------------------------
    assert np.array_equal(_np_con_indices(con_indices, np.int32(0)), np.array([0, 1, 2, 3, 4], dtype=np.int32))
    assert np.array_equal(_np_con_indices(con_indices, np.int32(1)), np.array([10, 11, 12, 13], dtype=np.int32))
    assert np.array_equal(_np_con_indices(con_indices, np.int32(2)), np.array([3, 11], dtype=np.int32))


@pytest.mark.parametrize(
    "i1_max,i2_max,i3_max,expected_result",
    [
        (4, 13, 11, 13),
        (4, 13, 30, 30),
        (40, 13, 30, 40),
    ],
)
def test_np_largest_con_index(i1_max: int, i2_max: int, i3_max: int, expected_result: int):
    # --- arrange -----------------------------------------
    _, con_indices = ConstraintList(
        [
            Constraint(int_set={0, 1, 2, 3, i1_max}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, i2_max}, min_count=0, max_count=7),
            Constraint(int_set={3, i3_max}, min_count=2, max_count=2),
        ]
    ).to_numpy()

    # --- act ---------------------------------------------
    result = _np_largest_con_index(con_indices)

    # --- assert ------------------------------------------
    assert result == expected_result


def test_np_con_total_violation():
    # --- arrange -----------------------------------------
    con_values = np.array(
        [
            [-7, 11],  # satisfied
            [0, 0],  # satisfied
            [3, 10],  # need 3 more
            [-30, -4],  # need 4 less
        ],
        dtype=np.int32,
    )

    # --- act ---------------------------------------------
    total_violation = _np_con_total_violation(con_values)

    # --- assert ------------------------------------------
    assert total_violation == 7
