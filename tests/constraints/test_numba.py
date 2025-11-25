from typing import Iterable

import numpy as np
import pytest

from max_div.constraints import Constraint, Constraints
from max_div.constraints._numba import (
    _build_array_repr,
    _np_con_build_index_sets,
    _np_con_indices,
    _np_con_max_value,
    _np_con_min_value,
    _np_con_satisfied,
)


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

    assert con_indices.shape[0] == 17  # (2*n_cons) + (5+4+2) = 6 + 11 = 17
    assert con_indices.dtype == np.int32

    for i, con in enumerate(cons):
        i_start = con_indices[2 * i]
        i_end = con_indices[2 * i + 1]
        assert list(con_indices[i_start:i_end]) == sorted(con.int_set)


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
    index_sets = _np_con_build_index_sets(con_indices, np.int32(2))

    # --- act ---------------------------------------------
    result = _np_con_satisfied(con_values, index_sets, np.array(list(samples), dtype=np.int32))

    # --- assert ------------------------------------------
    assert result == expected
