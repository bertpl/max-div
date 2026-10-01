import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance._metric import validate_vector_array_layout


# ==================================================================================================
#  The contract
# ==================================================================================================
def test_preprocess_follows_the_metrics_declaration(metric: DistanceMetric, vectors: np.ndarray):
    """A metric that does not preprocess gets its input back; one that does gets a new array."""
    # --- act --------------------------
    preprocessed = metric.preprocess(vectors)

    # --- assert -----------------------
    if metric.needs_preprocessed_vectors:
        assert not np.shares_memory(preprocessed, vectors)
    else:
        assert preprocessed is vectors


def test_preprocess_leaves_the_input_untouched(metric: DistanceMetric, vectors: np.ndarray):
    """Preprocessing never writes into the user's array, whichever metric asks."""
    # --- arrange ----------------------
    before = vectors.copy()

    # --- act --------------------------
    metric.preprocess(vectors)

    # --- assert -----------------------
    np.testing.assert_array_equal(vectors, before)


def test_preprocess_returns_the_layout_reads_expect(metric: DistanceMetric, vectors: np.ndarray):
    """What comes out is in the form every distance read expects, whether copied or not."""
    # --- act --------------------------
    preprocessed = metric.preprocess(vectors)

    # --- assert -----------------------
    validate_vector_array_layout(preprocessed)  # raises on violation


# ==================================================================================================
#  Cosine
# ==================================================================================================
def test_cosine_preprocessing_normalizes_rows(vectors: np.ndarray):
    """Cosine's preprocessed copy has unit-length rows."""
    # --- act --------------------------
    preprocessed = DistanceMetric.cosine().preprocess(vectors)

    # --- assert -----------------------
    np.testing.assert_allclose(np.linalg.norm(preprocessed, axis=1), 1.0, rtol=1e-6)


# ==================================================================================================
#  Precondition
# ==================================================================================================
@pytest.mark.parametrize(
    "bad_vectors",
    [
        np.zeros((4, 2), dtype=np.float64),  # wrong dtype
        np.asfortranarray(np.zeros((4, 2), dtype=np.float32)),  # wrong layout
        np.zeros(4, dtype=np.float32),  # wrong rank
    ],
    ids=["float64", "fortran", "1d"],
)
def test_validate_vector_array_layout_rejects_other_forms(bad_vectors: np.ndarray):
    """Anything but a 2D float32 C-contiguous array is refused, not converted."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="2D float32 C-contiguous"):
        validate_vector_array_layout(bad_vectors)


# ==================================================================================================
#  Along one axis
# ==================================================================================================
def test_along_axis_preprocessing_keeps_the_one_coordinate(vectors: np.ndarray):
    """The along-axis copy is an (n, 1) array holding exactly the requested coordinate."""
    # --- act --------------------------
    preprocessed = DistanceMetric.along_axis(2).preprocess(vectors)

    # --- assert -----------------------
    assert preprocessed.shape == (vectors.shape[0], 1)
    np.testing.assert_array_equal(preprocessed[:, 0], vectors[:, 2])


def test_along_axis_preprocessing_copies_a_single_column_too(vectors: np.ndarray):
    """Preprocessing copies the sliced coordinate, so a one-dimensional input yields a new array, not a view."""
    # --- arrange ----------------------
    single_column = np.ascontiguousarray(vectors[:, :1])

    # --- act --------------------------
    preprocessed = DistanceMetric.along_axis(0).preprocess(single_column)

    # --- assert -----------------------
    assert not np.shares_memory(preprocessed, single_column)
    np.testing.assert_array_equal(preprocessed, single_column)
