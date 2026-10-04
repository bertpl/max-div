import numpy as np

from max_div._core.distance_storage.allocation import InProcessDataMatrixAllocator, SharedMemoryDataMatrixAllocator
from max_div._core.distance_storage.data_matrix_producer import (
    AdoptingDataMatrixProducer,
    ComputingDataMatrixProducer,
    DataMatrixProducer,
)


def _vectors() -> np.ndarray:
    """Return a small float32 array for a producer to hand over."""
    return np.ascontiguousarray(np.random.default_rng(5).random((4, 3), dtype=np.float32))


def _counting_producer(calls: list[int]) -> ComputingDataMatrixProducer:
    """Return a producer of a 2x2 matrix of ones that records each time it computes."""

    def _fill_with_ones(buffer: np.ndarray) -> None:
        calls.append(1)
        buffer[:] = 1.0

    return ComputingDataMatrixProducer((2, 2), _fill_with_ones)


# ==================================================================================================
#  AdoptingDataMatrixProducer
# ==================================================================================================
def test_adopting_producer_hands_over_the_array_itself_in_process():
    """In this process, an adopted array is the data matrix itself."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act / assert -----------------
    assert AdoptingDataMatrixProducer(vectors).produce(0, InProcessDataMatrixAllocator()) is vectors


def test_adopting_producer_is_copied_into_shared_memory_under_its_id():
    """In shared memory, an adopted array is copied into a segment that is published under its matrix id."""
    # --- arrange ----------------------
    vectors = _vectors()
    allocator = SharedMemoryDataMatrixAllocator()

    # --- act --------------------------
    copied = np.array(AdoptingDataMatrixProducer(vectors).produce(3, allocator))
    published_ids = list(allocator.published_matrix_records.records)
    allocator.close()

    # --- assert -----------------------
    np.testing.assert_array_equal(copied, vectors)
    assert published_ids == [3]


# ==================================================================================================
#  ComputingDataMatrixProducer
# ==================================================================================================
def test_computing_producer_computes_into_the_buffer_that_the_allocator_allocates():
    """A computing producer fills a buffer of its shape and returns that buffer, so the matrix is never copied."""
    # --- arrange ----------------------
    filled_buffers = []

    def _fill_with_sevens(buffer: np.ndarray) -> None:
        buffer[:] = 7.0
        filled_buffers.append(buffer)

    # --- act --------------------------
    matrix = ComputingDataMatrixProducer((2, 3), _fill_with_sevens).produce(1, InProcessDataMatrixAllocator())

    # --- assert -----------------------
    assert len(filled_buffers) == 1
    assert filled_buffers[0] is matrix
    np.testing.assert_array_equal(matrix, np.full((2, 3), 7.0, dtype=np.float32))


# ==================================================================================================
#  DataMatrixProducer.produce_all
# ==================================================================================================
def test_produce_all_produces_each_data_matrix_once_under_its_id():
    """Every producer runs once, and its data matrix is returned under its matrix id."""
    # --- arrange ----------------------
    vectors = _vectors()
    calls: list[int] = []

    # --- act --------------------------
    matrices = DataMatrixProducer.produce_all(
        {0: AdoptingDataMatrixProducer(vectors), 2: _counting_producer(calls)}, InProcessDataMatrixAllocator()
    )

    # --- assert -----------------------
    assert sorted(matrices) == [0, 2]
    assert matrices[0] is vectors
    np.testing.assert_array_equal(matrices[2], np.ones((2, 2), dtype=np.float32))
    assert calls == [1]
