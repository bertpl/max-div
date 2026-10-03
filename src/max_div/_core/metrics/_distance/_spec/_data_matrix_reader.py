"""A data matrix reader returns the data matrices of one solve by their matrix id."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


# ==================================================================================================
#  DataMatrixReader
# ==================================================================================================
class DataMatrixReader(ABC):
    """A data matrix reader returns each data matrix of one solve by its matrix id.

    A data matrix is an array that a distance store reads: a full distance matrix, or the input
    vectors of a lazy distance store.  Each data matrix of a solve has an
    integer matrix id, which a `DistanceSpec` names.
    """

    @abstractmethod
    def array(self, matrix_id: int) -> NDArray[np.float32]:
        """Return the data matrix with the given id.

        The distance store keeps the returned array without copying it, so the array must stay valid while
        the store is in use, and must already have the layout that `DistanceStore.full_matrix` or
        `DistanceStore.lazy` requires.
        """
