import multiprocessing
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from max_div._core.distance_storage.allocation import PublishedDataMatrixRecord, PublishedDataMatrixRecords
from max_div._core.distance_storage.data_matrix_producer import AdoptingDataMatrixProducer
from max_div._core.distance_storage.data_matrix_publisher import SharedMemoryDataMatrixPublisher
from max_div._core.distance_storage.data_matrix_readers import InProcessDataMatrixReader, SharedMemoryDataMatrixReader
from max_div._core.metrics._distance import (
    DataMatrixReader,
    DistanceMetric,
    DistanceSpec,
    DistanceStore,
    FullMatrixDistanceSpec,
    VectorDistanceSpec,
    compute_full_matrix,
    get_distance,
)

# how long a spawned child may take to boot an interpreter, import max_div and answer
_CHILD_TIMEOUT_S = 120

_N = 12
_PAIRS = [(0, 1), (2, 7), (5, 5), (11, 3)]
# the exponent 3 must reach the worker unchanged; a worker without it would compute a different distance
_METRIC = DistanceMetric.minkowski(3)

# one spec per kind of store, over the data matrices that `_published_matrix_records` publishes
_DISTANCE_SPECS: tuple[DistanceSpec, ...] = (
    FullMatrixDistanceSpec(matrix_id=1, label=_METRIC.label),
    VectorDistanceSpec(matrix_id=0, metric=_METRIC, is_matrix_preprocessed=True),
)


def _vectors() -> np.ndarray:
    """Return the vectors that every distance store in this module derives its distances from."""
    return np.ascontiguousarray(np.random.default_rng(7).random((_N, 3), dtype=np.float32))


def _producers() -> dict[int, AdoptingDataMatrixProducer]:
    """Return the producers of the vectors (id 0) and of their full distance matrix (id 1)."""
    vectors = _vectors()
    return {
        0: AdoptingDataMatrixProducer(vectors),
        1: AdoptingDataMatrixProducer(compute_full_matrix(vectors, _METRIC)),
    }


@contextmanager
def _published_matrix_records() -> Iterator[PublishedDataMatrixRecords]:
    """Publish the data matrices of `_producers` in shared memory, for the duration of the block."""
    with SharedMemoryDataMatrixPublisher(_producers()) as published_matrix_records:
        yield published_matrix_records


def _read_pairs(store: DistanceStore) -> list[float]:
    """Return the distances that the store reports for a fixed set of pairs, self-pair included."""
    return [float(get_distance(store, np.int32(i), np.int32(j))) for i, j in _PAIRS]


def _expected_pairs() -> list[list[float]]:
    """Return the pairs read through unshared stores, in the order of `_DISTANCE_SPECS`."""
    vectors = _vectors()
    return [
        _read_pairs(DistanceStore.full_matrix_from_vectors(vectors, _METRIC)),
        _read_pairs(DistanceStore.lazy_from_vectors(vectors, _METRIC)),
    ]


# the child re-imports this module by name, so its entry point must be at module level
def _read_in_child(published_matrix_records: PublishedDataMatrixRecords, queue: multiprocessing.Queue) -> None:
    """Attach to the published data matrices and report the distances read through each spec's store."""
    with SharedMemoryDataMatrixReader(published_matrix_records) as reader:
        queue.put([_read_pairs(spec.build_distance_store(reader)) for spec in _DISTANCE_SPECS])


# ==================================================================================================
#  InProcessDataMatrixReader
# ==================================================================================================
def test_in_process_reader_returns_each_data_matrix_by_its_id():
    """An in-process reader is a data matrix reader that returns each produced data matrix under its id."""
    # --- arrange ----------------------
    producers = _producers()

    # --- act --------------------------
    reader = InProcessDataMatrixReader(producers)

    # --- assert -----------------------
    assert isinstance(reader, DataMatrixReader)
    assert reader.array(0) is producers[0].array
    assert reader.array(1) is producers[1].array


def test_in_process_reader_builds_the_stores_of_the_specs():
    """The stores that the specs build over an in-process reader read the same distances as unshared stores."""
    # --- act --------------------------
    reader = InProcessDataMatrixReader(_producers())

    # --- assert -----------------------
    assert [_read_pairs(spec.build_distance_store(reader)) for spec in _DISTANCE_SPECS] == _expected_pairs()


# ==================================================================================================
#  SharedMemoryDataMatrixReader
# ==================================================================================================
def test_spawned_process_reads_the_published_values():
    """A spawned process that reads the published data matrices builds stores with exactly the unshared distances."""
    # --- arrange ----------------------
    context = multiprocessing.get_context("spawn")  # never fork: numba's threading layer is fork-unsafe
    queue = context.Queue()

    # --- act --------------------------
    with _published_matrix_records() as published_matrix_records:
        child = context.Process(target=_read_in_child, args=(published_matrix_records, queue))
        child.start()
        try:
            actual = queue.get(timeout=_CHILD_TIMEOUT_S)
        finally:
            child.join(timeout=_CHILD_TIMEOUT_S)

    # --- assert -----------------------
    assert actual == _expected_pairs()


def test_shared_memory_reader_builds_the_stores_of_the_specs():
    """Reading the published data matrices in this process reproduces the unshared distances, for both kinds."""
    # --- arrange / act ----------------
    with (
        _published_matrix_records() as published_matrix_records,
        SharedMemoryDataMatrixReader(published_matrix_records) as reader,
    ):
        read = [_read_pairs(spec.build_distance_store(reader)) for spec in _DISTANCE_SPECS]

    # --- assert -----------------------
    assert read == _expected_pairs()


def test_stores_over_a_shared_memory_reader_cannot_be_written_through():
    """Nothing that is reachable from a store over a shared-memory segment can write into the segment."""
    # --- arrange / act ----------------
    with (
        _published_matrix_records() as published_matrix_records,
        SharedMemoryDataMatrixReader(published_matrix_records) as reader,
    ):
        stores = [spec.build_distance_store(reader) for spec in _DISTANCE_SPECS]
        writeable = [(store.matrix.flags.writeable, store.preprocessed_vectors.flags.writeable) for store in stores]

    # --- assert -----------------------
    assert writeable == [(False, False), (False, False)]


def test_leaving_a_shared_memory_reader_leaves_the_segments_usable():
    """Leaving a shared-memory reader releases only its mappings, so the segments stay for later readers."""
    # --- arrange / act ----------------
    with _published_matrix_records() as published_matrix_records:
        with SharedMemoryDataMatrixReader(published_matrix_records) as first:
            read_first = np.array(first.array(1))
        with SharedMemoryDataMatrixReader(published_matrix_records) as second:
            read_after = np.array(second.array(1))

    # --- assert -----------------------
    np.testing.assert_array_equal(read_first, read_after)


def test_reading_a_missing_segment_raises():
    """A record whose segment is gone cannot be read; the segments attached before it are closed."""
    # --- arrange ----------------------
    with _published_matrix_records() as published_matrix_records:
        missing = PublishedDataMatrixRecords(
            {
                **published_matrix_records.records,
                7: PublishedDataMatrixRecord(segment_name="max_div_missing_segment", shape=(2, 2)),
            }
        )

        # --- act / assert -------------
        with pytest.raises(FileNotFoundError), SharedMemoryDataMatrixReader(missing):
            pass
