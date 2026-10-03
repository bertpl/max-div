import pickle

import numpy as np
import pytest

from max_div._core.metrics._distance import (
    DataMatrixReader,
    DistanceMetric,
    DistanceSpec,
    DistanceStore,
    FullMatrixDistanceSpec,
    VectorDistanceSpec,
    compute_full_matrix,
)
from max_div._core.metrics._distance._store import KIND_FULL_MATRIX, KIND_LAZY
from tests._core.metrics._distance.helpers import all_pair_distances

# ==================================================================================================
#  Fixtures / helpers
# ==================================================================================================
L2 = DistanceMetric.l2_euclidean()
COSINE = DistanceMetric.cosine()

# the pickling test runs over this list, which holds one spec per subclass of `DistanceSpec`
_ONE_SPEC_PER_KIND = [
    FullMatrixDistanceSpec(matrix_id=4, label="user distances"),
    VectorDistanceSpec(matrix_id=2, metric=DistanceMetric.minkowski(3), is_matrix_preprocessed=True),
]


class _DictDataMatrixReader(DataMatrixReader):
    """A test reader returns the data matrices of a dict, keyed by matrix id."""

    def __init__(self, arrays: dict[int, np.ndarray]) -> None:
        """Keep the given data matrices, keyed by matrix id."""
        self._arrays = arrays

    def array(self, matrix_id: int) -> np.ndarray:
        """Return the data matrix with the given id from the dict."""
        return self._arrays[matrix_id]


# ==================================================================================================
#  Building a distance store
# ==================================================================================================
def test_full_matrix_spec_reads_the_distance_matrix_with_its_id(vectors: np.ndarray):
    """A full-matrix spec builds a full-matrix store over the matrix that its `matrix_id` names, not another one."""
    # --- arrange ----------------------
    matrix = compute_full_matrix(vectors, L2)
    reader = _DictDataMatrixReader({0: vectors, 3: matrix})

    # --- act --------------------------
    store = FullMatrixDistanceSpec(matrix_id=3, label="L2").build_distance_store(reader)

    # --- assert -----------------------
    assert store.kind == KIND_FULL_MATRIX
    assert np.shares_memory(store.matrix, matrix)


@pytest.mark.parametrize("metric", [L2, COSINE], ids=["non-preprocessing", "preprocessing"])
def test_preprocessed_vector_spec_computes_the_metrics_distances(metric: DistanceMetric, vectors: np.ndarray):
    """A vector spec over preprocessed vectors reports the same distances as a lazy store built from the raw vectors."""
    # --- arrange ----------------------
    reader = _DictDataMatrixReader({0: vectors, 2: metric.preprocess(vectors)})

    # --- act --------------------------
    store = VectorDistanceSpec(matrix_id=2, metric=metric, is_matrix_preprocessed=True).build_distance_store(reader)

    # --- assert -----------------------
    assert store.kind == KIND_LAZY
    assert all_pair_distances(store) == all_pair_distances(DistanceStore.lazy_from_vectors(vectors, metric))


def test_vector_spec_over_the_vectors_as_given_builds_no_store(vectors: np.ndarray):
    """A vector spec with `is_matrix_preprocessed` off builds no store, even for a metric that does not preprocess."""
    # --- arrange ----------------------
    reader = _DictDataMatrixReader({0: vectors})

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="preprocessed"):
        VectorDistanceSpec(matrix_id=0, metric=L2, is_matrix_preprocessed=False).build_distance_store(reader)


# ==================================================================================================
#  Label, equality, pickling
# ==================================================================================================
@pytest.mark.parametrize(
    "spec, expected",
    [
        (FullMatrixDistanceSpec(matrix_id=0, label="user distances"), "user distances"),
        (VectorDistanceSpec(matrix_id=0, metric=L2, is_matrix_preprocessed=False), "L2"),
        (VectorDistanceSpec(matrix_id=1, metric=DistanceMetric.along_axis(2), is_matrix_preprocessed=True), "axis 2"),
    ],
)
def test_label_names_the_distances(spec: DistanceSpec, expected: str):
    """A full-matrix spec's label is its field; a vector spec's label is its metric's."""
    # --- act / assert -----------------
    assert spec.label == expected


@pytest.mark.parametrize(
    "other, is_equal",
    [
        (VectorDistanceSpec(matrix_id=0, metric=L2, is_matrix_preprocessed=True), True),
        (VectorDistanceSpec(matrix_id=1, metric=L2, is_matrix_preprocessed=True), False),
        (VectorDistanceSpec(matrix_id=0, metric=L2, is_matrix_preprocessed=False), False),
        (VectorDistanceSpec(matrix_id=0, metric=COSINE, is_matrix_preprocessed=True), False),
        (FullMatrixDistanceSpec(matrix_id=0, label="L2"), False),
    ],
    ids=["same-fields", "other-matrix", "not-preprocessed", "other-metric", "other-kind"],
)
def test_specs_are_equal_when_kind_and_fields_are(other: DistanceSpec, is_equal: bool):
    """Specs compare and hash alike exactly when they have the same kind and the same fields."""
    # --- arrange ----------------------
    spec = VectorDistanceSpec(matrix_id=0, metric=L2, is_matrix_preprocessed=True)

    # --- act / assert -----------------
    assert (spec == other) is is_equal
    assert (len({spec, other}) == 1) is is_equal


@pytest.mark.parametrize("spec", _ONE_SPEC_PER_KIND, ids=["full-matrix", "vector"])
def test_spec_survives_pickling(spec: DistanceSpec):
    """A spec is picklable, which lets it reach a spawned worker inside an objective."""
    # --- act / assert -----------------
    assert pickle.loads(pickle.dumps(spec)) == spec  # noqa: S301 -- our own spec, not untrusted input


def test_one_spec_per_kind_covers_every_kind():
    """The pickling test covers every kind of spec."""
    # --- act / assert -----------------
    # `dataclass(slots=True)` replaces each subclass by a new class, and the replaced class stays listed among
    # the subclasses, so the kinds are compared by name
    assert {type(spec).__name__ for spec in _ONE_SPEC_PER_KIND} == {
        cls.__name__ for cls in DistanceSpec.__subclasses__()
    }


def test_the_base_spec_cannot_be_created():
    """Only the subclasses of `DistanceSpec` build a distance store, so the base class itself cannot be created."""
    # --- act / assert -----------------
    with pytest.raises(TypeError):
        DistanceSpec(matrix_id=0)
