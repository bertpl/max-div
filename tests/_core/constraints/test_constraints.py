import numpy as np
import pytest

from max_div._core.constraints.constraints import (
    Constraint,
    ConstraintList,
    _build_array_repr,
    _np_con_count_satisfied,
    _np_con_indices,
    _np_con_max_value,
    _np_con_membership,
    _np_con_min_value,
    _np_con_total_violation,
    _np_con_total_weighted_violation,
    _np_largest_con_index,
    to_numpy_membership,
)


def test_constraint_default_weight():
    # --- act --------------------------
    con = Constraint(int_set={0, 1}, min_count=1, max_count=2)

    # --- assert -----------------------
    assert con.weight == 1.0


@pytest.mark.parametrize("weight", [0, 0.0, -1.0, -0.001])
def test_constraint_rejects_non_positive_weight(weight: float):
    # --- act & assert -----------------
    with pytest.raises(ValueError, match="weight must be > 0"):
        Constraint(int_set={0, 1}, min_count=1, max_count=2, weight=weight)


def test_build_array_repr():
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
        Constraint(int_set={3, 11}, min_count=2, max_count=2),
    ]

    # --- act --------------------------
    con_values_1, con_indices_1 = _build_array_repr(cons)
    con_values_2, con_indices_2 = ConstraintList(cons).to_numpy()

    # --- assert -----------------------
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


def test_to_numpy_membership():
    """The packed membership array inverts con_indices: per item, the sorted ids of its constraints."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
        Constraint(int_set={3, 11}, min_count=2, max_count=2),
    ]
    n = 14
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

    # --- act --------------------------
    _, con_indices = _build_array_repr(cons)
    con_membership = to_numpy_membership(con_indices, m=len(cons), n=n)

    # --- assert -----------------------
    assert con_membership.dtype == np.int32
    assert con_membership.shape[0] == 2 * n + 11  # header + one payload entry per (constraint, item) pair
    for idx, expected_ids in expected_membership.items():
        assert list(_np_con_membership(con_membership, idx)) == expected_ids
    # segments are contiguous and in item order, so the payload region is exactly covered
    assert con_membership[0] == 2 * n
    assert con_membership[2 * n - 1] == con_membership.shape[0]


def test_to_numpy_membership_accepts_numpy_index():
    """The accessor takes np.int32 indices, as handed to it by SolverState's mutation methods."""
    # --- arrange ----------------------
    cons = [Constraint(int_set={0, 2}, min_count=1, max_count=2)]

    # --- act --------------------------
    _, con_indices = _build_array_repr(cons)
    con_membership = to_numpy_membership(con_indices, m=1, n=3)

    # --- assert -----------------------
    assert list(_np_con_membership(con_membership, np.int32(2))) == [0]
    assert list(_np_con_membership(con_membership, np.int32(1))) == []


def test_np_con_min_value():
    # --- arrange ----------------------
    con_values, _con_indices = ConstraintList(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    ).to_numpy()

    # --- act & assert -----------------
    assert _np_con_min_value(con_values, np.int32(0)) == 2
    assert _np_con_min_value(con_values, np.int32(1)) == 0
    assert _np_con_min_value(con_values, np.int32(2)) == 2


def test_np_con_max_value():
    # --- arrange ----------------------
    con_values, _con_indices = ConstraintList(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    ).to_numpy()

    # --- act & assert -----------------
    assert _np_con_max_value(con_values, np.int32(0)) == 3
    assert _np_con_max_value(con_values, np.int32(1)) == 7
    assert _np_con_max_value(con_values, np.int32(2)) == 2


def test_np_con_indices():
    # --- arrange ----------------------
    _con_values, con_indices = ConstraintList(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ]
    ).to_numpy()

    # --- act & assert -----------------
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
    # --- arrange ----------------------
    _, con_indices = ConstraintList(
        [
            Constraint(int_set={0, 1, 2, 3, i1_max}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, i2_max}, min_count=0, max_count=7),
            Constraint(int_set={3, i3_max}, min_count=2, max_count=2),
        ]
    ).to_numpy()

    # --- act --------------------------
    result = _np_largest_con_index(con_indices)

    # --- assert -----------------------
    assert result == expected_result


def test_np_con_total_violation():
    # --- arrange ----------------------
    con_values = np.array(
        [
            [-7, 11],  # satisfied
            [0, 0],  # satisfied
            [3, 10],  # need 3 more
            [-30, -4],  # need 4 less
        ],
        dtype=np.int32,
    )

    # --- act --------------------------
    total_violation = _np_con_total_violation(con_values)

    # --- assert -----------------------
    assert total_violation == 7


@pytest.mark.parametrize(
    "weights,quadratic,expected",
    [
        # per-constraint violations for the con_values below are v = [0, 0, 3, 4]
        ([1.0, 1.0, 1.0, 1.0], False, 7.0),  # Σ v            = 3 + 4
        ([1.0, 1.0, 1.0, 1.0], True, 25.0),  # Σ v²           = 9 + 16
        ([1.0, 1.0, 2.0, 0.5], False, 8.0),  # Σ w·v          = 2·3 + 0.5·4
        ([1.0, 1.0, 2.0, 0.5], True, 26.0),  # Σ w·v²         = 2·9 + 0.5·16
    ],
)
def test_np_con_total_weighted_violation(weights: list[float], quadratic: bool, expected: float):
    # --- arrange ----------------------
    con_values = np.array(
        [
            [-7, 11],  # satisfied            -> v = 0
            [0, 0],  # satisfied            -> v = 0
            [3, 10],  # need 3 more          -> v = 3
            [-30, -4],  # need 4 less          -> v = 4
        ],
        dtype=np.int32,
    )
    con_weights = np.array(weights, dtype=np.float32)

    # --- act --------------------------
    total = _np_con_total_weighted_violation(con_values, con_weights, quadratic)

    # --- assert -----------------------
    assert total == pytest.approx(expected)


@pytest.mark.parametrize(
    "con_values",
    [
        [[-7, 11], [0, 0], [3, 10], [-30, -4]],
        [[2, 3], [2, 3]],
        [[0, 0]],
        [[-1, -1], [5, 9]],
    ],
)
def test_np_con_total_weighted_violation_matches_fast_path(con_values: list[list[int]]):
    """Regression guard: unit weights + linear must reproduce the integer fast path exactly."""
    # --- arrange ----------------------
    cv = np.array(con_values, dtype=np.int32)
    unit_weights = np.ones(cv.shape[0], dtype=np.float32)

    # --- act --------------------------
    general = _np_con_total_weighted_violation(cv, unit_weights, False)
    fast = _np_con_total_violation(cv)

    # --- assert -----------------------
    assert float(general) == float(fast)


def test_np_con_count_satisfied():
    # --- arrange ----------------------
    con_values = np.array(
        [
            [-7, 11],  # satisfied (min_remaining <= 0, max_remaining >= 0)
            [0, 0],  # satisfied (boundary case)
            [3, 10],  # not satisfied (min_remaining > 0)
            [-30, -4],  # not satisfied (max_remaining < 0)
        ],
        dtype=np.int32,
    )

    # --- act --------------------------
    n_satisfied = _np_con_count_satisfied(con_values)

    # --- assert -----------------------
    assert n_satisfied == 2


def test_np_con_count_satisfied_empty():
    # --- arrange ----------------------
    con_values = np.zeros((0, 2), dtype=np.int32)

    # --- act --------------------------
    n_satisfied = _np_con_count_satisfied(con_values)

    # --- assert -----------------------
    assert n_satisfied == 0
