import multiprocessing
import pickle
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from max_div._core.metrics._distance import KIND_FULL_MATRIX, KIND_LAZY, DistanceMetric, DistanceStore, get_distance
from max_div._core.solver._distance_storage import SharedStoreSpec, attached_distance_store
from max_div._core.solver._distance_storage.allocation import SharedMemoryDistanceStoreAllocator
from max_div._core.solver._distance_storage.shared_memory import _attach_without_registering

# how long a spawned child may take to boot an interpreter, import max_div and answer
_CHILD_TIMEOUT_S = 120

_N = 12
_PAIRS = [(0, 1), (2, 7), (5, 5), (11, 3)]
_METRIC = DistanceMetric.l2_euclidean()


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


def _publish(store: DistanceStore) -> tuple[SharedMemoryDistanceStoreAllocator, SharedStoreSpec]:
    """Copy a distance store's array into a segment and return the allocator that owns it, with the spec."""
    allocator = SharedMemoryDistanceStoreAllocator()
    if store.kind == KIND_FULL_MATRIX:
        allocator.adopt(store.matrix, KIND_FULL_MATRIX, None)
    else:
        allocator.adopt(store.preprocessed_vectors, KIND_LAZY, _METRIC)
    return allocator, allocator.specs[0]


def _read_pairs(store: DistanceStore) -> list[float]:
    """Return the distances that the store reports for a fixed set of pairs, self-pair included."""
    return [float(get_distance(store, np.int32(i), np.int32(j))) for i, j in _PAIRS]


# the child re-imports this module by name, so its entry point must be at module level
def _read_in_child(specs: dict[str, SharedStoreSpec], queue: multiprocessing.Queue) -> None:
    """Attach to each published distance store and report the distances read through it."""
    results = {}
    for kind, spec in specs.items():
        with attached_distance_store(spec) as store:
            results[kind] = _read_pairs(store)
    queue.put(results)


# =================================================================================================
#  Round trip
# =================================================================================================
def test_spawned_process_reads_the_published_values():
    """A spawned process that attaches to a published distance store reads exactly what the unshared store holds."""
    # --- arrange ----------------------
    references = _reference_stores()
    expected = {kind: _read_pairs(store) for kind, store in references.items()}
    published = {kind: _publish(store) for kind, store in references.items()}
    context = multiprocessing.get_context("spawn")  # never fork: numba's threading layer is fork-unsafe
    queue = context.Queue()

    # --- act --------------------------
    child = context.Process(target=_read_in_child, args=({k: spec for k, (_, spec) in published.items()}, queue))
    child.start()
    try:
        actual = queue.get(timeout=_CHILD_TIMEOUT_S)
    finally:
        child.join(timeout=_CHILD_TIMEOUT_S)
        for allocator, _ in published.values():
            allocator.close()

    # --- assert -----------------------
    assert actual == expected


@pytest.mark.parametrize("kind", ["full_matrix", "lazy"])
def test_attached_store_reads_the_published_values(kind: str):
    """Attaching in this process reproduces the unshared distance store's distances, for both kinds."""
    # --- arrange ----------------------
    reference = _reference_stores()[kind]
    allocator, spec = _publish(reference)

    # --- act --------------------------
    with attached_distance_store(spec) as attached:
        read_attached = _read_pairs(attached)
    allocator.close()

    # --- assert -----------------------
    assert read_attached == _read_pairs(reference)


@pytest.mark.parametrize("kind, expected_kind", [("full_matrix", KIND_FULL_MATRIX), ("lazy", KIND_LAZY)])
def test_published_spec_names_the_kind_it_holds(kind: str, expected_kind: np.int32):
    """The spec carries the kind selector, so an attaching process rebuilds a distance store of the same kind."""
    # --- arrange / act ----------------
    allocator, spec = _publish(_reference_stores()[kind])
    allocator.close()

    # --- assert -----------------------
    assert spec.kind == expected_kind
    assert spec.shape[0] == _N


def test_spec_survives_pickling():
    """The spec is picklable, which lets it reach a spawned worker as an argument."""
    # --- arrange ----------------------
    allocator, spec = _publish(_reference_stores()["full_matrix"])
    allocator.close()

    # --- act --------------------------
    restored = pickle.loads(pickle.dumps(spec))  # noqa: S301 -- our own spec, not untrusted input

    # --- assert -----------------------
    assert restored == spec


def test_attached_minkowski_store_reads_the_published_values():
    """The metric's exponent must survive publish and attach, or an attached reader computes another distance."""
    # --- arrange ----------------------
    rng = np.random.default_rng(20260829)
    vectors = rng.standard_normal((12, 4)).astype(np.float32)
    metric = DistanceMetric.minkowski(3)
    reference = DistanceStore.lazy_from_vectors(vectors, metric)
    allocator = SharedMemoryDistanceStoreAllocator()
    allocator.adopt(reference.preprocessed_vectors, KIND_LAZY, metric)

    # --- act --------------------------
    with attached_distance_store(allocator.specs[0]) as attached:
        read_attached = _read_pairs(attached)
    allocator.close()

    # --- assert -----------------------
    assert allocator.specs[0].metric_p == 3.0
    assert read_attached == _read_pairs(reference)


# =================================================================================================
#  Read-only enforcement
# =================================================================================================
@pytest.mark.parametrize("kind", ["full_matrix", "lazy"])
def test_attached_store_cannot_be_written_through(kind: str):
    """Nothing that is reachable from an attached distance store can write into the shared segment."""
    # --- arrange ----------------------
    allocator, spec = _publish(_reference_stores()[kind])

    # --- act --------------------------
    with attached_distance_store(spec) as attached:
        matrix_writeable = attached.matrix.flags.writeable
        vectors_writeable = attached.preprocessed_vectors.flags.writeable
    allocator.close()

    # --- assert -----------------------
    assert not matrix_writeable
    assert not vectors_writeable


# =================================================================================================
#  Segment lifetime
# =================================================================================================
def test_attaching_leaves_the_segment_usable():
    """Closing an attachment releases only that mapping, so the segment survives for later readers."""
    # --- arrange ----------------------
    allocator, spec = _publish(_reference_stores()["full_matrix"])
    expected = _read_pairs(_reference_stores()["full_matrix"])

    # --- act --------------------------
    with attached_distance_store(spec) as first:
        read_first = _read_pairs(first)
    with attached_distance_store(spec) as second:
        read_after = _read_pairs(second)
    allocator.close()

    # --- assert -----------------------
    assert read_first == expected
    assert read_after == expected


def test_closing_the_allocator_destroys_the_segment():
    """The allocator's close unlinks the segment, so its name no longer resolves."""
    # --- arrange ----------------------
    allocator, spec = _publish(_reference_stores()["full_matrix"])

    # --- act --------------------------
    allocator.close()

    # --- assert -----------------------
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=spec.segment_name).close()


def test_attaching_without_registering_reads_and_leaves_the_owner_tracked():
    """The pre-3.13 attach path maps the segment and leaves the creating process's tracker entry intact."""
    # --- arrange ----------------------
    allocator, spec = _publish(_reference_stores()["full_matrix"])
    registered = resource_tracker.register
    expected = _reference_stores()["full_matrix"].matrix

    # --- act --------------------------
    segment = _attach_without_registering(spec.segment_name)
    values = np.ndarray(spec.shape, dtype=np.float32, buffer=segment.buf).copy()
    segment.close()
    allocator.close()  # the allocator still holds its own registration, so unlinking stays clean

    # --- assert -----------------------
    np.testing.assert_array_equal(values, expected)
    assert resource_tracker.register is registered  # the suppression lasted one constructor call
