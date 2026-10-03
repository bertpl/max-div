"""A data matrix reader returns the data matrices of one solve by their matrix id."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


# ==================================================================================================
#  DataMatrixReader
# ==================================================================================================
class DataMatrixReader(ABC):
    """A data matrix reader returns each data matrix of one solve by its matrix id.

    A data matrix is an array that a distance store reads: a full distance matrix, or the vectors
    that a lazy distance store computes its distances from.  Each data matrix of a solve has an
    integer matrix id, which a `DistanceSpec` names.
    """

    @abstractmethod
    def array(self, matrix_id: int) -> NDArray[np.float32]:
        """Return the data matrix with the given id."""
