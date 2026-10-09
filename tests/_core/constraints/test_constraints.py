import dataclasses

import numpy as np
import pytest

from max_div._core.constraints.constraints import (
    Constraint,
    _np_con_count_satisfied,
    _np_con_indices,
    _np_con_max_value,
    _np_con_membership,
    _np_con_min_value,
    _np_con_total_violation,
    _np_con_total_weighted_violation,
    _np_largest_con_index,
    to_numpy_constraints,
    to_numpy_membership,
)


def test_constraint_default_weight():
    # --- act --------------------------
    con = Constraint(int_set={0, 1}, min_count=1, max_count=2)

    # --- assert -----------------------
    assert con.weight == 1.0


@pytest.mark.parametrize("weight", [0, 0.0, -1.0, -0.001, float("nan"), float("inf"), float("-inf")])
def test_constraint_rejects_a_weight_that_is_not_finite_and_positive(weight: float):
    """A zero, negative, NaN or infinite weight raises at construction."""
    # --- act & assert -----------------
    with pytest.raises(ValueError, match="weight must be finite and > 0"):
        Constraint(int_set={0, 1}, min_count=1, max_count=2, weight=weight)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"int_set": set(), "min_count": 1, "max_count": 2}, "must not be empty"),
        ({"int_set": {0.5, 1}, "min_count": 1, "max_count": 2}, "must be integers"),
        ({"int_set": {True, 2}, "min_count": 1, "max_count": 2}, "must be integers"),
        ({"int_set": {-1, 1}, "min_count": 1, "max_count": 2}, "must be >= 0"),
        ({"int_set": {0, 1}, "min_count": 1.5, "max_count": 2}, "min_count must be an integer"),
        ({"int_set": {0, 1}, "min_count": True, "max_count": 2}, "min_count must be an integer"),
        ({"int_set": {0, 1}, "min_count": 1, "max_count": float("inf")}, "max_count must be an integer"),
        ({"int_set": {0, 1}, "min_count": 1, "max_count": None}, "max_count must be an integer"),
        ({"int_set": {0, 1}, "min_count": -1, "max_count": 2}, "min_count must be >= 0"),
        ({"int_set": {0, 1}, "min_count": 3, "max_count": 1}, "must be >= min_count"),
        ({"int_set": {0, 1}, "min_count": 3, "max_count": 3}, "must not exceed the number of distinct"),
        ({"int_set": [0, 0, 1], "min_count": 3, "max_count": 3}, "must not exceed the number of distinct"),
    ],
    ids=[
        "empty-int_set",
        "non-integer-member",
        "bool-member",
        "negative-member",
        "float-min_count",
        "bool-min_count",
        "infinite-max_count",
        "none-max_count",
        "negative-min_count",
        "min-above-max",
        "min-above-set-size",
        "min-above-distinct-members",
    ],
)
def test_constraint_rejects_invalid_definitions(kwargs: dict, match: str):
    """Every malformed constraint definition raises at construction; none reaches compiled code."""
    # --- act & assert -----------------
    with pytest.raises(ValueError, match=match):
        Constraint(**kwargs)


@pytest.mark.parametrize(
    "int_set",
    [
        {0, 1, 2},
        [2, 0, 1, 0],
        (0, 1, 2),
        range(3),
        np.array([0, 1, 2]),
        set(np.array([0, 1, 2], dtype=np.int32)),
    ],
    ids=["set", "list-with-repeats", "tuple", "range", "numpy-array", "set-of-numpy-integers"],
)
def test_constraint_stores_any_iterable_of_integers_as_a_frozenset_of_ints(int_set):
    """Any accepted iterable of integers, numpy integers included, is stored as a frozenset of plain ints.

    Repeated members are dropped.
    """
    # --- act --------------------------
    con = Constraint(int_set=int_set, min_count=1, max_count=2)

    # --- assert -----------------------
    assert isinstance(con.int_set, frozenset)
    assert con.int_set == frozenset({0, 1, 2})
    assert all(type(member) is int for member in con.int_set)


@pytest.mark.parametrize("field", [field.name for field in dataclasses.fields(Constraint)])
def test_constraint_is_immutable(field: str):
    """A field cannot change after construction, so a validated constraint stays valid."""
    # --- arrange ----------------------
    con = Constraint(int_set={0, 1, 2}, min_count=1, max_count=2)

    # --- act & assert -----------------
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(con, field, -1)


def test_constraint_min_equal_max_is_valid():
    """Coinciding bounds are an exact-count constraint, not an inverted one."""
    # --- act --------------------------
    con = Constraint(int_set={0, 1, 2}, min_count=2, max_count=2)

    # --- assert -----------------------
    assert (con.min_count, con.max_count) == (2, 2)


def test_to_numpy_constraints():
    """con_values holds each min_count and max_count; con_indices holds a 2m-element header of offsets.

    Each constraint's sorted members follow the header.
    """
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
        Constraint(int_set={3, 11}, min_count=2, max_count=2),
    ]

    # --- act --------------------------
    con_values, con_indices = to_numpy_constraints(cons, n=14)

    # --- assert -----------------------
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


@pytest.mark.parametrize(
    "max_count, expected_packed",
    [(5, 5), (10, 10), (11, 10), (2**40, 10)],
    ids=["below-n", "equal-to-n", "just-above-n", "beyond-int32"],
)
def test_to_numpy_constraints_clips_max_count_to_n(max_count: int, expected_packed: int):
    """A max_count above n is stored as n in con_values.

    The clipped value fits in int32 and does not change which selections satisfy the constraint.
    """
    # --- arrange ----------------------
    cons = [Constraint(int_set={0, 1, 2}, min_count=1, max_count=max_count)]

    # --- act --------------------------
    con_values, _ = to_numpy_constraints(cons, n=10)

    # --- assert -----------------------
    assert con_values[0, 1] == expected_packed
    assert cons[0].max_count == max_count


def test_to_numpy_constraints_rejects_an_index_of_n_or_more():
    """An index outside [0, n) raises before it can reach compiled code, where bounds are not checked."""
    # --- arrange ----------------------
    cons = [Constraint(int_set={0, 1}, min_count=1, max_count=1), Constraint(int_set={2, 10}, min_count=1, max_count=1)]

    # --- act & assert -----------------
    with pytest.raises(ValueError, match="Constraint 1 references item index 10"):
        to_numpy_constraints(cons, n=10)


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
    _, con_indices = to_numpy_constraints(cons, n)
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
    _, con_indices = to_numpy_constraints(cons, n=3)
    con_membership = to_numpy_membership(con_indices, m=1, n=3)

    # --- assert -----------------------
    assert list(_np_con_membership(con_membership, np.int32(2))) == [0]
    assert list(_np_con_membership(con_membership, np.int32(1))) == []


def test_np_con_min_value():
    # --- arrange ----------------------
    con_values, _con_indices = to_numpy_constraints(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ],
        n=50,
    )

    # --- act & assert -----------------
    assert _np_con_min_value(con_values, np.int32(0)) == 2
    assert _np_con_min_value(con_values, np.int32(1)) == 0
    assert _np_con_min_value(con_values, np.int32(2)) == 2


def test_np_con_max_value():
    # --- arrange ----------------------
    con_values, _con_indices = to_numpy_constraints(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ],
        n=50,
    )

    # --- act & assert -----------------
    assert _np_con_max_value(con_values, np.int32(0)) == 3
    assert _np_con_max_value(con_values, np.int32(1)) == 7
    assert _np_con_max_value(con_values, np.int32(2)) == 2


def test_np_con_indices():
    # --- arrange ----------------------
    _con_values, con_indices = to_numpy_constraints(
        [
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
            Constraint(int_set={3, 11}, min_count=2, max_count=2),
        ],
        n=50,
    )

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
    _, con_indices = to_numpy_constraints(
        [
            Constraint(int_set={0, 1, 2, 3, i1_max}, min_count=2, max_count=3),
            Constraint(int_set={10, 11, 12, i2_max}, min_count=0, max_count=7),
            Constraint(int_set={3, i3_max}, min_count=2, max_count=2),
        ],
        n=50,
    )

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


def test_np_largest_con_index_skips_an_empty_segment():
    """An empty constraint segment in a raw packed array is skipped, not read at [-1].

    `Constraint` rejects empty sets, so this can only arise from directly built arrays.
    """
    # --- arrange ----------------------
    # 2 constraints: first empty (start == end), second holding {4, 7}
    con_indices = np.array([4, 4, 4, 6, 4, 7], dtype=np.int32)

    # --- act & assert -----------------
    assert _np_largest_con_index(con_indices) == 7
