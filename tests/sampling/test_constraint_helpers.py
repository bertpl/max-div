from typing import Iterable

import numpy as np
import pytest

from max_div.sampling._constraint_helpers import (
    _build_array_repr,
    _build_con_membership,
    _is_int_in_sorted_array,
    _np_con_build_index_sets,
    _np_con_indices,
    _np_con_max_value,
    _np_con_min_value,
    _np_con_satisfied,
    _np_con_total_violation,
)
from max_div.solver._constraints import Constraint, Constraints


def test_build_array_repr():
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
        Constraint(int_set={3, 11}, min_count=2, max_count=2),
    ]

    # --- act ---------------------------------------------
    con_values, con_indices = _build_array_repr(cons)

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

    assert con_indices.shape[0] == 17  # (2*m) + (5+4+2) = 6 + 11 = 17
    assert con_indices.dtype == np.int32

    for i, con in enumerate(cons):
        i_start = con_indices[2 * i]
        i_end = con_indices[2 * i + 1]
        assert list(con_indices[i_start:i_end]) == sorted(con.int_set)


def test_build_con_membership():
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
        Constraint(int_set={3, 11}, min_count=2, max_count=2),
    ]
    m = np.int32(14)

    # --- act ---------------------------------------------
    con_membership = _build_con_membership(m, cons)

    # --- assert ------------------------------------------
    expected_membership = {
        0: [0],
        1: [0],
        2: [0],
        3: [0, 2],
        4: [0],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [1],
        11: [1, 2],
        12: [1],
        13: [1],
    }

    assert con_membership == expected_membership


def test_np_con_min_value():
    # --- arrange -----------------------------------------
    con_values, con_indices = _build_array_repr(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    )

    # --- act & assert ------------------------------------
    assert _np_con_min_value(con_values, np.int32(0)) == 2
    assert _np_con_min_value(con_values, np.int32(1)) == 0
    assert _np_con_min_value(con_values, np.int32(2)) == 2


def test_np_con_max_value():
    # --- arrange -----------------------------------------
    con_values, con_indices = _build_array_repr(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    )

    # --- act & assert ------------------------------------
    assert _np_con_max_value(con_values, np.int32(0)) == 3
    assert _np_con_max_value(con_values, np.int32(1)) == 7
    assert _np_con_max_value(con_values, np.int32(2)) == 2


def test_np_con_indices():
    # --- arrange -----------------------------------------
    con_values, con_indices = _build_array_repr(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    )

    # --- act & assert ------------------------------------
    assert np.array_equal(_np_con_indices(con_indices, np.int32(0)), np.array([0, 1, 2, 3, 4], dtype=np.int32))
    assert np.array_equal(_np_con_indices(con_indices, np.int32(1)), np.array([10, 11, 12, 13], dtype=np.int32))
    assert np.array_equal(_np_con_indices(con_indices, np.int32(2)), np.array([3, 11], dtype=np.int32))


def test_np_con_build_index_sets():
    # --- arrange -----------------------------------------
    con_values, con_indices = _build_array_repr(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    )

    # --- act ---------------------------------------------
    index_sets = _np_con_build_index_sets(con_indices, np.int32(3))

    # --- assert ------------------------------------------
    assert len(index_sets) == 3
    assert set(index_sets[0]) == {0, 1, 2, 3, 4}
    assert set(index_sets[1]) == {10, 11, 12, 13}
    assert set(index_sets[2]) == {3, 11}


@pytest.mark.parametrize(
    "samples, expected",
    [
        ([], False),
        ([1], False),
        ([0, 2], True),
        ([1, 3, 4], True),
        ([0, 2, 4, 11], True),
        ([0, 2, 4, 1], False),
        ([10, 11, 12, 13], False),
        ([10, 11], False),
    ],
)
def test_np_con_satisfied(samples: Iterable[int], expected: bool):
    # --- arrange -----------------------------------------
    cons = Constraints()
    cons.add(indices={0, 1, 2, 3, 4}, min_count=2, max_count=3)
    cons.add(indices={10, 11, 12, 13}, min_count=0, max_count=3)

    con_values, con_indices = cons.to_numpy()

    # --- act ---------------------------------------------
    result = _np_con_satisfied(con_values, con_indices, np.array(list(samples), dtype=np.int32))

    # --- assert ------------------------------------------
    assert result == expected


@pytest.mark.parametrize(
    "arr, value, expected",
    [
        ([1, 2, 10], 0, False),
        ([1, 2, 10], 1, True),
        ([1, 2, 10], 2, True),
        ([1, 2, 10], 5, False),
        ([1, 2, 10], 10, True),
        ([1, 2, 10], 20, False),
        (list(range(0, 1000, 2)), -1, False),
        (list(range(0, 1000, 2)), 0, True),
        (list(range(0, 1000, 2)), 77, False),
        (list(range(0, 1000, 2)), 998, True),
        (list(range(0, 1000, 2)), 1235, False),
    ],
)
def test_is_int_in_sorted_array(arr: list, value: int, expected: bool):
    assert (
        _is_int_in_sorted_array(
            arr=np.array(arr, dtype=np.int32),
            value=np.int32(value),
        )
        == expected
    )


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
