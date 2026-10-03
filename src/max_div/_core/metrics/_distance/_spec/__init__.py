"""The spec layer says which data matrix a set of distances is read from, and builds the distance store that reads it.

A `DistanceSpec` names a data matrix by its matrix id and holds no array: when a distance store is
built, the spec fetches its data matrix from a `DataMatrixReader`.
"""

from ._data_matrix_reader import DataMatrixReader
from ._distance_spec import DistanceSpec, PrecomputedDistanceSpec, VectorDistanceSpec

__all__ = [
    "DataMatrixReader",
    "DistanceSpec",
    "PrecomputedDistanceSpec",
    "VectorDistanceSpec",
]
