import numpy as np

from max_div._core.distance_storage.allocation import InProcessDataMatrixAllocator, SharedMemoryDataMatrixAllocator
from max_div._core.distance_storage.data_matrix_source import ComputedDataMatrixSource, ExistingDataMatrixSource


def _vectors() -> np.ndarray:
    """Return a small float32 array for a source to hold."""
    return np.ascontiguousarray(np.random.default_rng(5).random((4, 3), dtype=np.float32))


# ==================================================================================================
#  ExistingDataMatrixSource
# ==================================================================================================
def test_existing_source_is_adopted_without_a_copy_in_process():
    """In this process, an existing array is the data matrix itself."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act / assert -----------------
    assert ExistingDataMatrixSource(vectors).produce(0, InProcessDataMatrixAllocator()) is vectors


def test_existing_source_is_copied_into_shared_memory_under_its_id():
    """In shared memory, an existing array is copied into a segment that is published under its matrix id."""
    # --- arrange ----------------------
    vectors = _vectors()
    allocator = SharedMemoryDataMatrixAllocator()

    # --- act --------------------------
    copied = np.array(ExistingDataMatrixSource(vectors).produce(3, allocator))
    published_ids = list(allocator.published.matrices)
    allocator.close()

    # --- assert -----------------------
    np.testing.assert_array_equal(copied, vectors)
    assert published_ids == [3]


# ==================================================================================================
#  ComputedDataMatrixSource
# ==================================================================================================
def test_computed_source_computes_into_the_buffer_that_the_allocator_allocates():
    """A computed source fills a buffer of its shape and returns that buffer, so the matrix is never copied."""
    # --- arrange ----------------------
    filled_buffers = []

    def _fill_with_sevens(buffer: np.ndarray) -> None:
        buffer[:] = 7.0
        filled_buffers.append(buffer)

    # --- act --------------------------
    matrix = ComputedDataMatrixSource((2, 3), _fill_with_sevens).produce(1, InProcessDataMatrixAllocator())

    # --- assert -----------------------
    assert len(filled_buffers) == 1
    assert filled_buffers[0] is matrix
    np.testing.assert_array_equal(matrix, np.full((2, 3), 7.0, dtype=np.float32))
