import numpy as np
import pytest

from max_div._core.metrics._distance import DataMatrixReader


def test_a_reader_must_implement_array():
    """A reader that does not implement `array` cannot be created, and one that does returns its matrices by id."""

    # --- arrange ----------------------
    class _Incomplete(DataMatrixReader):
        """A reader that leaves `array` unimplemented."""

    class _Complete(DataMatrixReader):
        """A reader that returns a 2x2 matrix filled with the requested id."""

        def array(self, matrix_id: int) -> np.ndarray:
            """Return a 2x2 matrix filled with `matrix_id`."""
            return np.full((2, 2), matrix_id, dtype=np.float32)

    # --- act / assert -----------------
    with pytest.raises(TypeError):
        _Incomplete()
    assert _Complete().array(3)[0, 0] == 3.0
