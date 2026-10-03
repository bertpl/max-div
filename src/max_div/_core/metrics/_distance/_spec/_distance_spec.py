"""A distance spec says which data matrix a set of pairwise distances is read from, and how.

There are 2 kinds of spec:

- `PrecomputedDistanceSpec` reads the distances from a full distance matrix;
- `VectorDistanceSpec` computes each distance on demand from vectors, with its distance metric.

A spec is a small frozen value that holds no array, so it can travel to a worker process inside
a pickled objective.  Two specs are equal when they have the same kind and the same fields;
whether a distance matrix holds exactly the distances of some vector spec is not visible from
the specs, so such a pair compares unequal.
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
    """A distance spec names the data matrix that a set of distances is read from, and builds the store that reads it."""

    matrix_id: int  # the id of the data matrix that the distances are read from

    @property
    @abstractmethod
    def label(self) -> str:
        """Return the short name of these distances, as an objective's label shows it."""

    @abstractmethod
    def distance_store(self, data_matrices: DataMatrixReader) -> DistanceStore:
        """Return the distance store that reads this spec's data matrix, which it fetches from `data_matrices` by its id."""


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
    - **on:** the vectors as `DistanceMetric.preprocess` returns them for the metric, the form that
      a lazy distance store reads.

    A distance store is built only from a spec with the flag on.  A metric that preprocesses cannot
    read the user's vectors as given, and refusing every spec with the flag off keeps one rule for
    every metric, including one whose preprocessing returns the vectors unchanged.
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
