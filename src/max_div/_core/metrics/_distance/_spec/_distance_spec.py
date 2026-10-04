"""A distance spec identifies, by its id, the data matrix of a set of pairwise distances, and how to read them.

A data matrix is a full distance matrix or a set of vectors, and each subclass of `DistanceSpec`
reads one of the two.

A spec is a small frozen value that holds no array, so it can be pickled inside an objective and
sent to a worker process.

Specs are equal when they have the same kind and the same fields.  A full-matrix spec and a vector
spec always compare unequal, even when the full-matrix spec's distance matrix holds exactly the
vector spec's distances, because a spec holds no array to compare.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from max_div._core.metrics._distance._metric import DistanceMetric
from max_div._core.metrics._distance._store import DistanceStore

from ._data_matrix_reader import DataMatrixReader


# ==================================================================================================
#  DistanceSpec
# ==================================================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class DistanceSpec(ABC):
    """A distance spec names the data matrix of a set of distances, and builds the distance store over that matrix."""

    # This id names the data matrix of the distances.
    matrix_id: int

    @property
    @abstractmethod
    def label(self) -> str:
        """Return the short name of these distances, which an objective shows in its own label."""

    @abstractmethod
    def build_distance_store(self, data_matrix_reader: DataMatrixReader) -> DistanceStore:
        """Return the distance store over this spec's data matrix, fetched from `data_matrix_reader` by `matrix_id`."""


# ==================================================================================================
#  Kinds of spec
# ==================================================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class FullMatrixDistanceSpec(DistanceSpec):
    """A full-matrix distance spec reads its distances from a full (n, n) distance matrix.

    The label is a field, because a distance matrix carries no name of its own.
    """

    # `slots=True` turns this field into a slot attribute on the generated class, and that attribute
    # overrides the abstract `label` property inherited from `DistanceSpec`.  The explicit `field()`
    # keeps the field required: without it, the dataclass would take the inherited property object as
    # the field's default.
    label: str = field()

    def build_distance_store(self, data_matrix_reader: DataMatrixReader) -> DistanceStore:
        """Return the full-matrix distance store over this spec's distance matrix."""
        return DistanceStore.full_matrix(data_matrix_reader.array(self.matrix_id))


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorDistanceSpec(DistanceSpec):
    """A vector distance spec computes its distances on demand from vectors, with its distance metric.

    `is_matrix_preprocessed` says which vectors `matrix_id` names:

    - **off:** the user's vectors as given;
    - **on:** the vectors as `DistanceMetric.preprocess` returns them for the metric, which are the
      input vectors of a lazy distance store.

    `build_distance_store` raises for every spec with the flag off.  For a metric that preprocesses, a lazy
    distance store over the user's vectors as given would compute wrong distances.  Raising for every
    metric, including a metric whose preprocessing returns the vectors unchanged, means a caller never
    needs to know which metrics preprocess.

    A spec with the flag off still has a label and compares by its fields, so it can name a set of
    distances before its vectors are preprocessed.
    """

    metric: DistanceMetric
    is_matrix_preprocessed: bool

    @property
    def label(self) -> str:
        """Return the metric's label, e.g. `L2` or `axis 2`."""
        return self.metric.label

    def build_distance_store(self, data_matrix_reader: DataMatrixReader) -> DistanceStore:
        """Return the lazy distance store that computes this spec's distances from its preprocessed vectors.

        Raises:
            ValueError: If `is_matrix_preprocessed` is off.
        """
        if not self.is_matrix_preprocessed:
            raise ValueError(
                f"{self!r} names the user's vectors as given; a distance store reads only vectors that are "
                "preprocessed for the metric."
            )
        return DistanceStore.lazy(data_matrix_reader.array(self.matrix_id), self.metric)
