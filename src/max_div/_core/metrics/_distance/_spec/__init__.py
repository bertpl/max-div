"""The spec layer identifies the data matrix of a set of distances, and builds the distance store over that matrix.

A data matrix is a full distance matrix or a set of vectors.
"""

from ._data_matrix_reader import DataMatrixReader
from ._distance_spec import DistanceSpec, PrecomputedDistanceSpec, VectorDistanceSpec

__all__ = [
    "DataMatrixReader",
    "DistanceSpec",
    "PrecomputedDistanceSpec",
    "VectorDistanceSpec",
]
