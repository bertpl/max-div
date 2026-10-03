import pickle

import numpy as np
import pytest

from max_div._core.metrics._distance import (
    DataMatrixReader,
    DistanceMetric,
    DistanceSpec,
    DistanceStore,
    PrecomputedDistanceSpec,
    VectorDistanceSpec,
    compute_full_matrix,
    get_distance,
)
from max_div._core.metrics._distance._store import KIND_FULL_MATRIX, KIND_LAZY

# ==================================================================================================
#  Fixtures / helpers
# ==================================================================================================
L2 = DistanceMetric.l2_euclidean()
COSINE = DistanceMetric.cosine()


class _DictDataMatrixReader(DataMatrixReader):
    """Return the data matrices of a dict, keyed by matrix id."""

    def __init__(self, arrays: dict[int, np.ndarray]) -> None:
        self._arrays = arrays

    def array(self, matrix_id: int) -> np.ndarray:
        return self._arrays[matrix_id]


def _vectors() -> np.ndarray:
    """Return 6 float32 vectors with no zero row, so that every metric can read them."""
    return np.random.default_rng(20261003).random((6, 3), dtype=np.float32) + 0.1


def _all_pairs(store: DistanceStore) -> list[float]:
    """Return every (i, j) distance the store reports, self-pairs included."""
    return [float(get_distance(store, np.int32(i), np.int32(j))) for i in range(store.n) for j in range(store.n)]


# ==================================================================================================
#  Building a distance store
# ==================================================================================================
def test_precomputed_spec_reads_the_distance_matrix_with_its_id():
    """A precomputed spec builds a full-matrix store over its own matrix id, not over another matrix of the reader."""
    # --- arrange ----------------------
    matrix = compute_full_matrix(_vectors(), L2)
    reader = _DictDataMatrixReader({0: _vectors(), 3: matrix})

    # --- act --------------------------
    store = PrecomputedDistanceSpec(matrix_id=3, label="L2").distance_store(reader)

    # --- assert -----------------------
    assert store.kind == KIND_FULL_MATRIX
    assert np.shares_memory(store.matrix, matrix)


@pytest.mark.parametrize("metric", [L2, COSINE], ids=["non-preprocessing", "preprocessing"])
def test_preprocessed_vector_spec_computes_the_metrics_distances(metric: DistanceMetric):
    """A vector spec over preprocessed vectors builds a lazy store that reads the same distances as the metric."""
    # --- arrange ----------------------
    vectors = _vectors()
    reader = _DictDataMatrixReader({0: vectors, 2: metric.preprocess(vectors)})

    # --- act --------------------------
    store = VectorDistanceSpec(matrix_id=2, metric=metric, is_preprocessed=True).distance_store(reader)

    # --- assert -----------------------
    assert store.kind == KIND_LAZY
    assert _all_pairs(store) == _all_pairs(DistanceStore.lazy_from_vectors(vectors, metric))


def test_vector_spec_over_the_vectors_as_given_builds_no_store():
    """A vector spec with `is_preprocessed` off refuses to build a store, even for a metric that does not preprocess."""
    # --- arrange ----------------------
    reader = _DictDataMatrixReader({0: _vectors()})

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="preprocessed"):
        VectorDistanceSpec(matrix_id=0, metric=L2, is_preprocessed=False).distance_store(reader)


# ==================================================================================================
#  Label, equality, pickling
# ==================================================================================================
@pytest.mark.parametrize(
    "spec, expected",
    [
        (PrecomputedDistanceSpec(matrix_id=0, label="user distances"), "user distances"),
        (VectorDistanceSpec(matrix_id=0, metric=L2, is_preprocessed=False), "L2"),
        (VectorDistanceSpec(matrix_id=1, metric=DistanceMetric.along_axis(2), is_preprocessed=True), "axis 2"),
    ],
)
def test_label_names_the_distances(spec: DistanceSpec, expected: str):
    """A precomputed spec's label is its field; a vector spec's label is its metric's."""
    # --- act / assert -----------------
    assert spec.label == expected


@pytest.mark.parametrize(
    "other, is_equal",
    [
        (VectorDistanceSpec(matrix_id=0, metric=L2, is_preprocessed=True), True),
        (VectorDistanceSpec(matrix_id=1, metric=L2, is_preprocessed=True), False),
        (VectorDistanceSpec(matrix_id=0, metric=L2, is_preprocessed=False), False),
        (VectorDistanceSpec(matrix_id=0, metric=COSINE, is_preprocessed=True), False),
        (PrecomputedDistanceSpec(matrix_id=0, label="L2"), False),
    ],
    ids=["same-fields", "other-matrix", "other-phase", "other-metric", "other-kind"],
)
def test_specs_are_equal_when_kind_and_fields_are(other: DistanceSpec, is_equal: bool):
    """Two specs compare and hash alike exactly when they have the same kind and the same fields."""
    # --- arrange ----------------------
    spec = VectorDistanceSpec(matrix_id=0, metric=L2, is_preprocessed=True)

    # --- act / assert -----------------
    assert (spec == other) is is_equal
    assert (len({spec, other}) == 1) is is_equal


@pytest.mark.parametrize(
    "spec",
    [
        PrecomputedDistanceSpec(matrix_id=4, label="user distances"),
        VectorDistanceSpec(matrix_id=2, metric=DistanceMetric.minkowski(3), is_preprocessed=True),
    ],
    ids=["precomputed", "vector"],
)
def test_spec_survives_pickling(spec: DistanceSpec):
    """A spec is picklable, which lets it reach a spawned worker inside an objective."""
    # --- act / assert -----------------
    assert pickle.loads(pickle.dumps(spec)) == spec  # noqa: S301 -- our own spec, not untrusted input


def test_the_base_spec_cannot_be_created():
    """Only the kinds of spec build a store, so the base class itself cannot be created."""
    # --- act / assert -----------------
    with pytest.raises(TypeError):
        DistanceSpec(matrix_id=0)
