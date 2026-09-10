import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance._metric import preprocess_vectors, validate_vector_array_layout


def _vectors() -> np.ndarray:
    """Return a small float32 C-contiguous array with no zero rows, so every metric accepts it."""
    return np.ascontiguousarray(np.random.default_rng(7).random((6, 3), dtype=np.float32) + 0.1)


# =================================================================================================
#  The contract
# =================================================================================================
def test_preprocess_follows_the_metrics_declaration(metric: DistanceMetric):
    """A metric that does not preprocess gets its input back; one that does gets a new array."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act --------------------------
    preprocessed = preprocess_vectors(vectors, metric)

    # --- assert -----------------------
    if metric.needs_preprocessed_vectors:
        assert not np.shares_memory(preprocessed, vectors)
    else:
        assert preprocessed is vectors


def test_preprocess_leaves_the_input_untouched(metric: DistanceMetric):
    """Preprocessing never writes into the user's array, whichever metric asks."""
    # --- arrange ----------------------
    vectors = _vectors()
    before = vectors.copy()

    # --- act --------------------------
    preprocess_vectors(vectors, metric)

    # --- assert -----------------------
    np.testing.assert_array_equal(vectors, before)


def test_preprocess_returns_the_layout_reads_expect(metric: DistanceMetric):
    """What comes out is in the form every distance read expects, whether copied or not."""
    # --- act --------------------------
    preprocessed = preprocess_vectors(_vectors(), metric)

    # --- assert -----------------------
    validate_vector_array_layout(preprocessed)  # raises on violation


# =================================================================================================
#  Cosine
# =================================================================================================
def test_cosine_preprocessing_normalizes_rows():
    """Cosine's preprocessed copy has unit-length rows."""
    # --- act --------------------------
    preprocessed = preprocess_vectors(_vectors(), DistanceMetric.cosine())

    # --- assert -----------------------
    np.testing.assert_allclose(np.linalg.norm(preprocessed, axis=1), 1.0, rtol=1e-6)


def test_cosine_preprocessing_rejects_zero_rows():
    """A zero row has no direction, so cosine preprocessing raises, naming the row."""
    # --- arrange ----------------------
    vectors = np.array([[1, 2], [0, 0], [3, 4]], dtype=np.float32)

    # --- act / assert -----------------
    with pytest.raises(ValueError, match=r"zero vector.*row 1"):
        preprocess_vectors(vectors, DistanceMetric.cosine())


# =================================================================================================
#  Precondition
# =================================================================================================
@pytest.mark.parametrize(
    "vectors",
    [
        np.zeros((4, 2), dtype=np.float64),  # wrong dtype
        np.asfortranarray(np.zeros((4, 2), dtype=np.float32)),  # wrong layout
        np.zeros(4, dtype=np.float32),  # wrong rank
    ],
    ids=["float64", "fortran", "1d"],
)
def test_validate_vector_array_layout_rejects_other_forms(vectors: np.ndarray):
    """Anything but a 2D float32 C-contiguous array is refused, not converted."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="2D float32 C-contiguous"):
        validate_vector_array_layout(vectors)


# =================================================================================================
#  Along one axis
# =================================================================================================
def test_along_axis_preprocessing_keeps_the_one_coordinate():
    """The along-axis copy is an (n, 1) array holding exactly the requested coordinate."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act --------------------------
    preprocessed = preprocess_vectors(vectors, DistanceMetric.along_axis(2))

    # --- assert -----------------------
    assert preprocessed.shape == (vectors.shape[0], 1)
    np.testing.assert_array_equal(preprocessed[:, 0], vectors[:, 2])


def test_along_axis_preprocessing_copies_a_single_column_too():
    """With one-dimensional vectors the slice would share memory, so only an explicit copy yields a new array."""
    # --- arrange ----------------------
    vectors = np.ascontiguousarray(_vectors()[:, :1])

    # --- act --------------------------
    preprocessed = preprocess_vectors(vectors, DistanceMetric.along_axis(0))

    # --- assert -----------------------
    assert not np.shares_memory(preprocessed, vectors)
    np.testing.assert_array_equal(preprocessed, vectors)


def test_along_axis_preprocessing_rejects_a_missing_coordinate():
    """An axis at or beyond the dimension count is refused."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="do not have"):
        preprocess_vectors(_vectors(), DistanceMetric.along_axis(3))
