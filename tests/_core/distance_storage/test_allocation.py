import numpy as np

from max_div._core.distance_storage.allocation import InProcessDataMatrixAllocator, SharedMemoryDataMatrixAllocator
from max_div._core.distance_storage.shared_memory import AttachedDataMatrixRegistry


def _vectors() -> np.ndarray:
    """Return a small float32 array to adopt."""
    return np.ascontiguousarray(np.random.default_rng(3).random((5, 2), dtype=np.float32))


# ==================================================================================================
#  InProcessDataMatrixAllocator
# ==================================================================================================
def test_in_process_allocate_returns_a_fresh_writable_buffer():
    """In this process, a buffer is a plain float32 numpy array that a source can fill."""
    # --- act --------------------------
    buffer = InProcessDataMatrixAllocator().allocate(1, (5, 5))

    # --- assert -----------------------
    assert buffer.shape == (5, 5)
    assert buffer.dtype == np.float32
    assert buffer.flags.writeable


def test_in_process_adopt_returns_the_array_itself():
    """In this process, adopting copies nothing: `adopt` returns the array that it was given."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act / assert -----------------
    assert InProcessDataMatrixAllocator().adopt(0, vectors) is vectors


# ==================================================================================================
#  SharedMemoryDataMatrixAllocator
# ==================================================================================================
def test_shared_allocator_publishes_one_segment_per_matrix_id():
    """Each data matrix lives in a segment of its own, published under its matrix id with its shape."""
    # --- arrange ----------------------
    allocator = SharedMemoryDataMatrixAllocator()

    # --- act --------------------------
    allocator.allocate(1, (3, 3))
    allocator.adopt(0, _vectors())
    published_matrices = allocator.published_matrices
    allocator.close()

    # --- assert -----------------------
    assert {matrix_id: matrix.shape for matrix_id, matrix in published_matrices.matrices.items()} == {
        1: (3, 3),
        0: (5, 2),
    }
    assert published_matrices.matrices[0].segment_name != published_matrices.matrices[1].segment_name


def test_shared_adopt_copies_the_array_into_its_segment():
    """Adopting copies the array into the segment, so the returned array holds the same values in other memory."""
    # --- arrange ----------------------
    allocator = SharedMemoryDataMatrixAllocator()
    vectors = _vectors()

    # --- act --------------------------
    adopted = allocator.adopt(0, vectors)
    copied = np.array(adopted)
    shares_memory = np.shares_memory(adopted, vectors)
    allocator.close()

    # --- assert -----------------------
    assert not shares_memory
    np.testing.assert_array_equal(copied, vectors)


def test_shared_published_matrix_reads_back_what_was_written():
    """Another reader that attaches to a published segment reads the bytes that were written into it."""
    # --- arrange ----------------------
    allocator = SharedMemoryDataMatrixAllocator()
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
    allocator.allocate(4, (3, 3))[:] = matrix

    # --- act --------------------------
    with AttachedDataMatrixRegistry(allocator.published_matrices) as reader:
        seen = np.array(reader.array(4))
    allocator.close()

    # --- assert -----------------------
    np.testing.assert_array_equal(seen, matrix)


def test_shared_degenerate_shape_still_gets_a_segment():
    """An empty array still gets a segment, because the operating system rejects a segment of zero bytes."""
    # --- arrange ----------------------
    allocator = SharedMemoryDataMatrixAllocator()

    # --- act --------------------------
    buffer = allocator.allocate(1, (0, 0))
    published_matrices = allocator.published_matrices
    allocator.close()

    # --- assert -----------------------
    assert buffer.size == 0
    assert published_matrices.matrices[1].shape == (0, 0)


def test_shared_close_forgets_its_segments():
    """A second close is harmless, and the published records stay available."""
    # --- arrange ----------------------
    allocator = SharedMemoryDataMatrixAllocator()
    allocator.allocate(1, (2, 2))

    # --- act --------------------------
    allocator.close()
    allocator.close()

    # --- assert -----------------------
    assert list(allocator.published_matrices.matrices) == [1]
