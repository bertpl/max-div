import numba
import numpy as np

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore, compute_full_matrix, get_distance
from max_div._core.metrics._distance._metric._custom_distance import address_of_custom_distance_function
from tests._core.metrics._distance.helpers import custom_l1
from tests.helpers import as_python_callable


@numba.njit
def _first_plus_twice_second(a: np.ndarray, b: np.ndarray) -> float:
    """Return a[0] + 2 * b[0], a distance that changes when its arguments swap."""
    return a[0] + 2.0 * b[0]


def test_the_address_is_a_positive_integer_that_stays_the_same():
    """Every call for one function returns the same positive address."""
    # --- act --------------------------
    addresses = {address_of_custom_distance_function(custom_l1) for _ in range(3)}

    # --- assert -----------------------
    (address,) = addresses
    assert address > 0


def test_a_lazy_store_reads_the_value_of_the_function():
    """A lazy store over a custom metric reads the function's return value, narrowed to float32."""
    # --- arrange ----------------------
    vectors = np.random.default_rng(20261014).random((5, 3)).astype(np.float32)
    store = DistanceStore.lazy_from_vectors(vectors, DistanceMetric.custom(custom_l1))

    # --- act --------------------------
    distance = as_python_callable(get_distance)(store, np.int32(1), np.int32(3))

    # --- assert -----------------------
    assert distance == np.float32(custom_l1(vectors[1], vectors[3]))


def test_the_function_is_called_with_the_lower_index_first():
    """(i, j) and (j, i) both read f(vector i, vector j) for i < j, so lazy and full-matrix distances agree."""
    # --- arrange ----------------------
    vectors = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    metric = DistanceMetric.custom(_first_plus_twice_second)
    store = DistanceStore.lazy_from_vectors(vectors, metric)

    # --- act --------------------------
    matrix = compute_full_matrix(vectors, metric)
    lazy_matrix = np.array(
        [[as_python_callable(get_distance)(store, np.int32(i), np.int32(j)) for j in range(4)] for i in range(4)]
    )

    # --- assert -----------------------
    expected = np.array(
        [[0.0 if i == j else vectors[min(i, j), 0] + 2.0 * vectors[max(i, j), 0] for j in range(4)] for i in range(4)],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(matrix, expected)
    np.testing.assert_array_equal(lazy_matrix, expected)
