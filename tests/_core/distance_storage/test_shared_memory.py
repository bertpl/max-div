import multiprocessing
import pickle
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from max_div._core.distance_storage.data_matrix_registry import DataMatrixRegistry
from max_div._core.distance_storage.data_matrix_source import ExistingDataMatrixSource
from max_div._core.distance_storage.shared_memory import (
    AttachedDataMatrixRegistry,
    PublishedDataMatrices,
    PublishedDataMatrix,
    PublishedDistanceStores,
)
from max_div._core.metrics._distance import (
    DistanceMetric,
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
_METRIC = DistanceMetric.minkowski(
    3
)  # the exponent 3 must reach the worker unchanged; a worker without it would compute a different distance


def _vectors() -> np.ndarray:
    """Return the vectors that every distance store in this module derives its distances from."""
    return np.ascontiguousarray(np.random.default_rng(7).random((_N, 3), dtype=np.float32))


def _reference_stores() -> dict[str, DistanceStore]:
    """Return one ordinary, unshared distance store per kind, as the values to reproduce."""
    vectors = _vectors()
    return {
        "full_matrix": DistanceStore.full_matrix_from_vectors(vectors, _METRIC),
        "lazy": DistanceStore.lazy_from_vectors(vectors, _METRIC),
    }


@contextmanager
def _published_distance_stores() -> Iterator[PublishedDistanceStores]:
    """Publish one distance store per kind, full matrix first, for the duration of the block."""
    vectors = _vectors()
    sources = {
        0: ExistingDataMatrixSource(vectors),
        1: ExistingDataMatrixSource(compute_full_matrix(vectors, _METRIC)),
    }
    distance_specs = (
        FullMatrixDistanceSpec(matrix_id=1, label=_METRIC.label),
        VectorDistanceSpec(matrix_id=0, metric=_METRIC, is_matrix_preprocessed=True),
    )
    with DataMatrixRegistry.publish_to_shared_memory(sources) as published_matrices:
        yield PublishedDistanceStores(published_matrices, distance_specs)


def _read_pairs(store: DistanceStore) -> list[float]:
    """Return the distances that the store reports for a fixed set of pairs, self-pair included."""
    return [float(get_distance(store, np.int32(i), np.int32(j))) for i, j in _PAIRS]


def _expected_pairs() -> list[list[float]]:
    """Return the pairs read through the unshared stores, in the order that `_published` publishes them."""
    references = _reference_stores()
    return [_read_pairs(references["full_matrix"]), _read_pairs(references["lazy"])]


# the child re-imports this module by name, so its entry point must be at module level
def _read_in_child(published_distance_stores: PublishedDistanceStores, queue: multiprocessing.Queue) -> None:
    """Attach to the published distance stores and report the distances read through each one."""
    with published_distance_stores.attached_distance_stores() as stores:
        queue.put([_read_pairs(store) for store in stores])


# ==================================================================================================
#  Round trip
# ==================================================================================================
def test_spawned_process_reads_the_published_values():
    """A spawned process that attaches to published distance stores reads exactly what the unshared stores hold."""
    # --- arrange ----------------------
    context = multiprocessing.get_context("spawn")  # never fork: numba's threading layer is fork-unsafe
    queue = context.Queue()

    # --- act --------------------------
    with _published_distance_stores() as published_distance_stores:
        child = context.Process(target=_read_in_child, args=(published_distance_stores, queue))
        child.start()
        try:
            actual = queue.get(timeout=_CHILD_TIMEOUT_S)
        finally:
            child.join(timeout=_CHILD_TIMEOUT_S)

    # --- assert -----------------------
    assert actual == _expected_pairs()


def test_attached_stores_read_the_published_values():
    """Attaching in this process reproduces the unshared stores' distances, for both kinds of store."""
    # --- arrange / act ----------------
    with (
        _published_distance_stores() as published_distance_stores,
        published_distance_stores.attached_distance_stores() as stores,
    ):
        read = [_read_pairs(store) for store in stores]

    # --- assert -----------------------
    assert read == _expected_pairs()


def test_published_stores_survive_pickling():
    """The published record is picklable, which lets it reach a spawned worker as an argument."""
    # --- arrange / act ----------------
    with _published_distance_stores() as published_distance_stores:
        restored = pickle.loads(pickle.dumps(published_distance_stores))  # noqa: S301 -- our own record, not untrusted input

    # --- assert -----------------------
    assert restored == published_distance_stores


# ==================================================================================================
#  Read-only enforcement
# ==================================================================================================
def test_attached_stores_cannot_be_written_through():
    """Nothing that is reachable from an attached distance store can write into the shared segment."""
    # --- arrange / act ----------------
    with (
        _published_distance_stores() as published_distance_stores,
        published_distance_stores.attached_distance_stores() as stores,
    ):
        writeable = [(store.matrix.flags.writeable, store.preprocessed_vectors.flags.writeable) for store in stores]

    # --- assert -----------------------
    assert writeable == [(False, False), (False, False)]


# ==================================================================================================
#  Segment lifetime
# ==================================================================================================
def test_attaching_leaves_the_segments_usable():
    """Leaving an attached registry releases only its mappings, so the segments survive for later readers."""
    # --- arrange / act ----------------
    with _published_distance_stores() as published_distance_stores:
        with published_distance_stores.attached_distance_stores() as first:
            read_first = [_read_pairs(store) for store in first]
        with published_distance_stores.attached_distance_stores() as second:
            read_after = [_read_pairs(store) for store in second]

    # --- assert -----------------------
    assert read_first == read_after == _expected_pairs()


def test_attaching_to_a_missing_segment_raises():
    """A published record whose segment is gone cannot be attached; the segments attached before it are closed."""
    # --- arrange ----------------------
    with _published_distance_stores() as published_distance_stores:
        live = published_distance_stores.published_matrices
        missing = PublishedDataMatrices(
            {**live.matrices, 7: PublishedDataMatrix(segment_name="max_div_missing_segment", shape=(2, 2))}
        )

        # --- act / assert -------------
        with pytest.raises(FileNotFoundError), AttachedDataMatrixRegistry(missing):
            pass
