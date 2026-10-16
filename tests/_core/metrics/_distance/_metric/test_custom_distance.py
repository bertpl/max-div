import numba
import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore, compute_full_matrix, get_distance
from max_div._core.metrics._distance._metric._custom_distance import BoundDistanceFunction
from tests._core.metrics._distance.helpers import custom_l1, custom_minkowski
from tests.helpers import as_directly_callable


@numba.njit
def _first_plus_twice_second(a: np.ndarray, b: np.ndarray) -> float:
    """Return a[0] + 2 * b[0], a distance that changes when its arguments swap."""
    return a[0] + 2.0 * b[0]


@numba.njit
def _with_one_value(a: np.ndarray, b: np.ndarray, value: float) -> float:
    """Return 0, for tests that need a function that takes 1 value after its 2 vectors."""
    return 0.0


# ==================================================================================================
#  BoundDistanceFunction
# ==================================================================================================
@pytest.mark.parametrize(
    "function, params", [(custom_l1, ()), (custom_minkowski, (3.0, True))], ids=["no values", "2 values"]
)
def test_equal_bound_functions_share_1_positive_address(function, params):
    """Bound functions created separately from one function and equal values are equal, with 1 positive address."""
    # --- act --------------------------
    bound_functions = [BoundDistanceFunction(function, params) for _ in range(3)]

    # --- assert -----------------------
    assert len(set(bound_functions)) == 1
    (address,) = {bound_function.address() for bound_function in bound_functions}
    assert address > 0


@pytest.mark.parametrize("params", [(3, True), (3.0, 1)], ids=["int power", "int flag"])
def test_a_value_of_another_type_makes_another_bound_function(params):
    """`3` equals `3.0` and `1` equals `True` in Python, but each gives a bound function and an address of its own."""
    # --- arrange ----------------------
    reference = BoundDistanceFunction(custom_minkowski, (3.0, True))

    # --- act --------------------------
    other = BoundDistanceFunction(custom_minkowski, params)

    # --- assert -----------------------
    assert other != reference
    assert other.address() != reference.address()


def test_a_bound_function_differs_from_anything_else():
    """A bound function never equals an object of another type, not even its own function."""
    # --- act / assert -----------------
    assert BoundDistanceFunction(custom_l1, ()) != custom_l1


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.True_, True),
        (np.int32(2), 2),
        (np.uint8(2), 2),
        (np.float32(0.5), 0.5),
        (np.float64(0.5), 0.5),
        (2**63 - 1, 2**63 - 1),
        (-(2**63), -(2**63)),
    ],
    ids=repr,
)
def test_each_value_becomes_a_python_bool_int_or_float(value, expected):
    """A numpy scalar becomes the Python scalar of its kind, so it equals a bound function given that Python scalar."""
    # --- act --------------------------
    bound_function = BoundDistanceFunction(_with_one_value, (value,))

    # --- assert -----------------------
    (converted,) = bound_function.params
    assert type(converted) is type(expected)
    assert converted == expected
    assert bound_function == BoundDistanceFunction(_with_one_value, (expected,))


@pytest.mark.parametrize("value", ["3", None, 1j, np.array([1.0])], ids=repr)
def test_a_value_that_is_not_a_bool_an_int_or_a_float_is_rejected(value):
    """A string, None, a complex number or an array as a value raises TypeError."""
    # --- act / assert -----------------
    with pytest.raises(TypeError, match="a bool, an int or a float"):
        BoundDistanceFunction(_with_one_value, (value,))


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1, np.uint64(2**63)], ids=repr)
def test_an_int_outside_int64_is_rejected(value):
    """An integer that numba cannot type as int64 raises ValueError when the bound function is created."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="int64"):
        BoundDistanceFunction(_with_one_value, (value,))


@pytest.mark.parametrize("params", [(3.0,), (3.0, True, 1)], ids=["too few", "too many"])
def test_values_that_do_not_fit_the_function_are_rejected(params):
    """Too few or too many values for the function's arguments after its 2 vectors raise TypeError."""
    # --- act / assert -----------------
    with pytest.raises(TypeError, match="argument"):
        BoundDistanceFunction(custom_minkowski, params)


@pytest.mark.parametrize(
    "function, params, expected",
    [(custom_l1, (), "custom_l1"), (custom_minkowski, (3.0, True), "custom_minkowski(3.0, True)")],
)
def test_the_name_is_the_function_name_followed_by_any_values(function, params, expected):
    """The name is the function's name, with the values' reprs in parentheses if there are any."""
    # --- act / assert -----------------
    assert BoundDistanceFunction(function, params).name == expected


# ==================================================================================================
#  Calling a custom distance function
# ==================================================================================================
@pytest.mark.parametrize(
    "metric, distance_of_rows",
    [
        (DistanceMetric.custom(custom_l1), custom_l1),
        (
            DistanceMetric.custom(custom_minkowski, params=(3.0, False)),
            lambda a, b: custom_minkowski(a, b, 3.0, False),
        ),
    ],
    ids=["no values", "2 values"],
)
def test_a_lazy_store_reads_the_value_of_the_function(metric, distance_of_rows):
    """A lazy store over a custom metric reads the function's return value for its values, narrowed to float32."""
    # --- arrange ----------------------
    vectors = np.random.default_rng(20261014).random((5, 3)).astype(np.float32)
    store = DistanceStore.lazy_from_vectors(vectors, metric)

    # --- act --------------------------
    distance = as_directly_callable(get_distance)(store, np.int32(1), np.int32(3))

    # --- assert -----------------------
    assert distance == np.float32(distance_of_rows(vectors[1], vectors[3]))


def test_the_function_is_called_with_the_lower_index_first():
    """(i, j) and (j, i) both read f(vector i, vector j) for i < j, so lazy and full-matrix distances agree."""
    # --- arrange ----------------------
    vectors = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    metric = DistanceMetric.custom(_first_plus_twice_second)
    store = DistanceStore.lazy_from_vectors(vectors, metric)

    # --- act --------------------------
    matrix = compute_full_matrix(vectors, metric)
    lazy_matrix = np.array(
        [[as_directly_callable(get_distance)(store, np.int32(i), np.int32(j)) for j in range(4)] for i in range(4)]
    )

    # --- assert -----------------------
    expected = np.array(
        [[0.0 if i == j else vectors[min(i, j), 0] + 2.0 * vectors[max(i, j), 0] for j in range(4)] for i in range(4)],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(matrix, expected)
    np.testing.assert_array_equal(lazy_matrix, expected)
