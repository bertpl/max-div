import numpy as np
import pytest

from max_div._core.metrics._distance._metric import validate_vector_array_layout


@pytest.mark.parametrize(
    "bad_vectors",
    [
        np.zeros((4, 2), dtype=np.float64),  # wrong dtype
        np.asfortranarray(np.zeros((4, 2), dtype=np.float32)),  # wrong layout
        np.zeros(4, dtype=np.float32),  # wrong rank
    ],
    ids=["float64", "fortran", "1d"],
)
def test_validate_vector_array_layout_rejects_other_forms(bad_vectors: np.ndarray):
    """Anything but a 2D float32 C-contiguous array is refused, not converted."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="2D float32 C-contiguous"):
        validate_vector_array_layout(bad_vectors)
