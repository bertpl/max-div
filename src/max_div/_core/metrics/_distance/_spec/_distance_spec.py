"""A distance spec names the data matrix of a set of pairwise distances, and says how to obtain the distances from it.

A data matrix is a full distance matrix or a set of vectors.  Each kind of spec is a subclass of
`DistanceSpec`:

- `PrecomputedDistanceSpec` reads the distances from a full distance matrix;
- `VectorDistanceSpec` computes each distance on demand from vectors, with its distance metric.

A spec is a small frozen value that holds no array, so it can be pickled inside an objective and
sent to a worker process.

Specs are equal when they have the same kind and the same fields.  A precomputed spec and a vector
spec always compare unequal, even when the distance matrix holds exactly the vector spec's
distances, because the specs alone cannot show that.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    def distance_store(self, data_matrices: DataMatrixReader) -> DistanceStore:
        """Return the distance store that reads this spec's data matrix, fetched from `data_matrices` by `matrix_id`."""


# ==================================================================================================
#  Kinds of spec
# ==================================================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class PrecomputedDistanceSpec(DistanceSpec):
    """A precomputed distance spec reads its distances from a full (n, n) distance matrix.

    The label is a field, because a distance matrix carries no name of its own.
    """

    # `slots=True` lets this field implement the abstract `label` property: the slot replaces the
    # inherited property on this class.
    label: str

    def distance_store(self, data_matrices: DataMatrixReader) -> DistanceStore:
        """Return the full-matrix distance store over this spec's distance matrix."""
        return DistanceStore.full_matrix(data_matrices.array(self.matrix_id))


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorDistanceSpec(DistanceSpec):
    """A vector distance spec computes its distances on demand from vectors, with its distance metric.

    `is_preprocessed` says which vectors `matrix_id` names:

    - **off:** the user's vectors as given;
    - **on:** the vectors as `DistanceMetric.preprocess` returns them for the metric, which are the
      vectors that a lazy distance store reads.

    `distance_store` raises for every spec with the flag off.  For a metric that preprocesses, a lazy
    distance store over the user's vectors as given would compute wrong distances; raising for every
    metric, including a metric whose preprocessing returns the vectors unchanged, keeps a single rule.

    A spec with the flag off still has a label and compares by its fields, so it can name a set of
    distances before its vectors are preprocessed.
    """

    metric: DistanceMetric
    is_preprocessed: bool

    @property
    def label(self) -> str:
        """Return the metric's label, e.g. `L2` or `axis 2`."""
        return self.metric.label

    def distance_store(self, data_matrices: DataMatrixReader) -> DistanceStore:
        """Return the lazy distance store that computes this spec's distances from its preprocessed vectors.

        Raises:
            ValueError: If `is_preprocessed` is off.
        """
        if not self.is_preprocessed:
            raise ValueError(
                f"{self!r} names the user's vectors as given; a distance store reads only vectors that are "
                "preprocessed for the metric."
            )
        return DistanceStore.lazy(data_matrices.array(self.matrix_id), self.metric)
