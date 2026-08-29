import multiprocessing
import pickle
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from max_div._core.metrics._distance import (
    KIND_CONDENSED,
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceMetric,
    DistanceStore,
    SharedDistanceStore,
    SharedStoreSpec,
    attached_distance_store,
    compute_pdist,
    get_distance,
    publish_distance_store,
)
from max_div._core.metrics._distance._shared_memory import _attach_without_registering

# how long a spawned child may take to boot an interpreter, import max_div and answer
_CHILD_TIMEOUT_S = 120

_N = 12
_PAIRS = [(0, 1), (2, 7), (5, 5), (11, 3)]


def _vectors() -> np.ndarray:
    """Return the vectors every backend in this module derives its distances from."""
    return np.ascontiguousarray(np.random.default_rng(7).random((_N, 3), dtype=np.float32))


def _reference_stores() -> dict[str, DistanceStore]:
    """Return one ordinary (unshared) store per backend, as the values to reproduce."""
    vectors = _vectors()
    return {
        "condensed": DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.l2_euclidean()), n=_N),
        "full_matrix": DistanceStore.full_matrix_from_vectors(vectors, DistanceMetric.l2_euclidean()),
        "lazy": DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean()),
    }


def _published(store: DistanceStore) -> SharedDistanceStore:
    """Return a shared-memory owner holding a copy of the given store's data."""
    return publish_distance_store(store, _N)


def _read_pairs(store: DistanceStore) -> list[float]:
    """Return the distances the store reports for a fixed set of pairs, self-pair included."""
    return [float(get_distance(store, np.int32(i), np.int32(j))) for i, j in _PAIRS]


# the child re-imports this module by name, so its entry point must be at module level
def _read_in_child(specs: dict[str, SharedStoreSpec], queue: multiprocessing.Queue) -> None:
    """Attach to each published store and report the distances read through it."""
    results = {}
    for backend, spec in specs.items():
        with attached_distance_store(spec) as store:
            results[backend] = _read_pairs(store)
    queue.put(results)


# =================================================================================================
#  Round trip
# =================================================================================================
def test_spawned_process_reads_the_published_values():
    """A spawned process attached to a published store reads exactly what an unshared store holds."""
    # --- arrange ----------------------
    references = _reference_stores()
    expected = {backend: _read_pairs(store) for backend, store in references.items()}
    published = {backend: _published(store) for backend, store in references.items()}
    context = multiprocessing.get_context("spawn")  # never fork: numba's threading layer is fork-unsafe
    queue = context.Queue()

    # --- act --------------------------
    child = context.Process(target=_read_in_child, args=({b: p.spec for b, p in published.items()}, queue))
    child.start()
    try:
        actual = queue.get(timeout=_CHILD_TIMEOUT_S)
    finally:
        child.join(timeout=_CHILD_TIMEOUT_S)
        for owner in published.values():
            owner.close()

    # --- assert -----------------------
    assert actual == expected


@pytest.mark.parametrize("backend", ["condensed", "full_matrix", "lazy"])
def test_attached_store_reads_the_published_values(backend: str):
    """Attaching in-process reproduces the unshared store's distances for every backend."""
    # --- arrange ----------------------
    reference = _reference_stores()[backend]

    # --- act --------------------------
    with _published(reference) as owner, attached_distance_store(owner.spec) as attached:
        read_attached = _read_pairs(attached)
        read_owner = _read_pairs(owner.store)

    # --- assert -----------------------
    assert read_attached == _read_pairs(reference)
    assert read_owner == _read_pairs(reference)


@pytest.mark.parametrize(
    "backend, kind",
    [("condensed", KIND_CONDENSED), ("full_matrix", KIND_FULL_MATRIX), ("lazy", KIND_LAZY)],
)
def test_published_spec_names_the_backend_it_holds(backend: str, kind: np.int32):
    """The spec carries the backend selector, so an attaching process rebuilds the same store."""
    # --- arrange / act ----------------
    with _published(_reference_stores()[backend]) as owner:
        spec = owner.spec

    # --- assert -----------------------
    assert spec.kind == kind
    assert spec.n == _N


def test_spec_survives_pickling():
    """The spec is picklable, which lets it reach a spawned worker as an argument."""
    # --- arrange / act ----------------
    with _published(_reference_stores()["condensed"]) as owner:
        restored = pickle.loads(pickle.dumps(owner.spec))  # noqa: S301 -- our own spec, not untrusted input

    # --- assert -----------------------
    # Compare field-wise, because the placeholder metric_p is NaN and NaN never compares equal to itself.
    assert restored._replace(metric_p=0.0) == owner.spec._replace(metric_p=0.0)
    assert np.isnan(restored.metric_p)
    assert np.isnan(owner.spec.metric_p)


# =================================================================================================
#  Read-only enforcement
# =================================================================================================
@pytest.mark.parametrize("backend", ["condensed", "full_matrix", "lazy"])
def test_attached_store_cannot_be_written_through(backend: str):
    """Nothing reachable from an attached store can write into the shared segment."""
    # --- arrange / act ----------------
    with _published(_reference_stores()[backend]) as owner, attached_distance_store(owner.spec) as attached:
        # --- assert -------------------
        assert not attached.pdist.flags.writeable
        assert not attached.matrix.flags.writeable
        assert not attached.vectors.flags.writeable


def test_publishing_does_not_copy_the_segment_into_the_store():
    """The published store reads the segment itself rather than a copy of it."""
    # --- arrange / act ----------------
    with _published(_reference_stores()["full_matrix"]) as owner:
        # --- assert -------------------
        assert np.shares_memory(owner.store.matrix, owner.buffer)


# =================================================================================================
#  Segment lifetime
# =================================================================================================
def test_attaching_leaves_the_segment_usable():
    """Closing an attachment releases only that mapping, so the segment survives for later readers."""
    # --- arrange ----------------------
    owner = _published(_reference_stores()["condensed"])
    expected = _read_pairs(_reference_stores()["condensed"])

    # --- act --------------------------
    with attached_distance_store(owner.spec) as first:
        read_first = _read_pairs(first)
    with attached_distance_store(owner.spec) as second:
        read_after = _read_pairs(second)
    owner.close()

    # --- assert -----------------------
    assert read_first == expected
    assert read_after == expected


def test_closing_the_owner_destroys_the_segment():
    """The owner's close unlinks the segment, so its name no longer resolves."""
    # --- arrange ----------------------
    owner = _published(_reference_stores()["condensed"])
    name = owner.spec.segment_name

    # --- act --------------------------
    owner.close()

    # --- assert -----------------------
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name).close()


def test_attaching_without_registering_reads_and_leaves_the_owner_tracked():
    """The pre-3.13 attach path maps the segment and leaves the publisher's tracker entry intact."""
    # --- arrange ----------------------
    owner = _published(_reference_stores()["condensed"])
    registered = resource_tracker.register
    expected = np.array(owner.buffer)

    # --- act --------------------------
    segment = _attach_without_registering(owner.spec.segment_name)
    values = np.ndarray(owner.spec.shape, dtype=np.float32, buffer=segment.buf).copy()
    segment.close()
    owner.close()  # the owner still holds its own registration, so unlinking stays clean

    # --- assert -----------------------
    np.testing.assert_array_equal(values, expected)
    assert resource_tracker.register is registered  # suppression lasted one constructor call


def test_degenerate_shape_still_claims_a_segment():
    """A store with no distances to hold still publishes, since the OS rejects a zero-size segment."""
    # --- arrange / act ----------------
    with SharedDistanceStore.allocate((0,), int(KIND_CONDENSED), n=1) as owner:
        # --- assert -------------------
        assert owner.store.n == np.int32(1)
        assert owner.buffer.size == 0
