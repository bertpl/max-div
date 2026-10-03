from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from max_div._core.distance_storage.data_matrix_registry import DataMatrixRegistry
from max_div._core.distance_storage.data_matrix_source import ComputedDataMatrixSource, ExistingDataMatrixSource
from max_div._core.distance_storage.shared_memory import SharedMemoryDataMatrixReader
from max_div._core.metrics._distance import DataMatrixReader


def _vectors() -> np.ndarray:
    """Return a small float32 array for a source to hold."""
    return np.ascontiguousarray(np.random.default_rng(11).random((4, 2), dtype=np.float32))


def _counting_source(calls: list[int]) -> ComputedDataMatrixSource:
    """Return a source of a 2x2 matrix of ones that records each time it is computed."""

    def _fill_with_ones(buffer: np.ndarray) -> None:
        calls.append(1)
        buffer[:] = 1.0

    return ComputedDataMatrixSource((2, 2), _fill_with_ones)


# ==================================================================================================
#  In process
# ==================================================================================================
def test_in_process_registry_returns_each_data_matrix_by_its_id():
    """A registry in this process is a data matrix reader that returns each produced data matrix under its id."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act --------------------------
    registry = DataMatrixRegistry.in_process({0: ExistingDataMatrixSource(vectors), 2: _counting_source([])})

    # --- assert -----------------------
    assert isinstance(registry, DataMatrixReader)
    assert registry.array(0) is vectors
    np.testing.assert_array_equal(registry.array(2), np.ones((2, 2), dtype=np.float32))


def test_registry_produces_each_data_matrix_once():
    """Each source is produced once, however many distance stores later read its data matrix."""
    # --- arrange ----------------------
    calls: list[int] = []
    registry = DataMatrixRegistry.in_process({1: _counting_source(calls)})

    # --- act --------------------------
    first, second = registry.array(1), registry.array(1)

    # --- assert -----------------------
    assert calls == [1]
    assert first is second


# ==================================================================================================
#  Shared memory
# ==================================================================================================
def test_publishing_to_shared_memory_records_every_data_matrix_under_its_id():
    """Publishing produces every data matrix in shared memory, and an attached reader returns it under its id."""
    # --- arrange ----------------------
    vectors = _vectors()
    sources = {0: ExistingDataMatrixSource(vectors), 5: _counting_source([])}

    # --- act --------------------------
    with (
        DataMatrixRegistry.publish_to_shared_memory(sources) as published_matrix_records,
        SharedMemoryDataMatrixReader(published_matrix_records) as reader,
    ):
        read_vectors, read_ones = np.array(reader.array(0)), np.array(reader.array(5))

    # --- assert -----------------------
    assert sorted(published_matrix_records.records) == [0, 5]
    np.testing.assert_array_equal(read_vectors, vectors)
    np.testing.assert_array_equal(read_ones, np.ones((2, 2), dtype=np.float32))


def test_leaving_the_published_block_destroys_the_segments():
    """After the block every segment is destroyed, so its name no longer resolves."""
    # --- arrange ----------------------
    with DataMatrixRegistry.publish_to_shared_memory(
        {0: ExistingDataMatrixSource(_vectors())}
    ) as published_matrix_records:
        segment_name = published_matrix_records.records[0].segment_name

    # --- act / assert -----------------
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=segment_name).close()
