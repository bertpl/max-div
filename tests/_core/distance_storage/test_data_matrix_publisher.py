from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from max_div._core.distance_storage.data_matrix_producer import (
    AdoptingDataMatrixProducer,
    ComputingDataMatrixProducer,
)
from max_div._core.distance_storage.data_matrix_publisher import SharedMemoryDataMatrixPublisher
from max_div._core.distance_storage.data_matrix_readers import SharedMemoryDataMatrixReader


def _vectors() -> np.ndarray:
    """Return a small float32 array for a producer to hand over."""
    return np.ascontiguousarray(np.random.default_rng(11).random((4, 2), dtype=np.float32))


def _fill_with_ones(buffer: np.ndarray) -> None:
    """Write ones into the whole buffer."""
    buffer[:] = 1.0


def _fail(buffer: np.ndarray) -> None:
    """Raise, as a producer that cannot compute its data matrix does."""
    raise RuntimeError("cannot compute this data matrix")


def test_publisher_publishes_every_data_matrix_under_its_id():
    """Publishing produces every data matrix in shared memory, and a reader finds each one under its id."""
    # --- arrange ----------------------
    vectors = _vectors()
    producers = {0: AdoptingDataMatrixProducer(vectors), 5: ComputingDataMatrixProducer((2, 2), _fill_with_ones)}

    # --- act --------------------------
    with (
        SharedMemoryDataMatrixPublisher(producers) as published_matrix_records,
        SharedMemoryDataMatrixReader(published_matrix_records) as reader,
    ):
        read_vectors, read_ones = np.array(reader.array(0)), np.array(reader.array(5))

    # --- assert -----------------------
    assert sorted(published_matrix_records.records) == [0, 5]
    np.testing.assert_array_equal(read_vectors, vectors)
    np.testing.assert_array_equal(read_ones, np.ones((2, 2), dtype=np.float32))


def test_leaving_the_publisher_destroys_the_segments():
    """After the block every segment is destroyed, so its name no longer resolves."""
    # --- arrange ----------------------
    with SharedMemoryDataMatrixPublisher({0: AdoptingDataMatrixProducer(_vectors())}) as published_matrix_records:
        segment_name = published_matrix_records.records[0].segment_name

    # --- act / assert -----------------
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=segment_name).close()


def test_a_failing_producer_propagates_its_error():
    """When a producer raises, the publisher raises the same error and no block runs."""
    # --- arrange ----------------------
    producers = {0: AdoptingDataMatrixProducer(_vectors()), 1: ComputingDataMatrixProducer((2, 2), _fail)}

    # --- act / assert -----------------
    with pytest.raises(RuntimeError, match="cannot compute"), SharedMemoryDataMatrixPublisher(producers):
        pass
