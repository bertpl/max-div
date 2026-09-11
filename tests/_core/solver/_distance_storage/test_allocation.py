import numpy as np

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import KIND_FULL_MATRIX, KIND_LAZY
from max_div._core.solver._distance_storage import attached_distance_store
from max_div._core.solver._distance_storage.allocation import (
    InProcessDistanceStoreAllocator,
    SharedMemoryDistanceStoreAllocator,
)

L2 = DistanceMetric.l2_euclidean()


def _vectors() -> np.ndarray:
    """Return a small float32 array to adopt."""
    return np.ascontiguousarray(np.random.default_rng(3).random((5, 2), dtype=np.float32))


# =================================================================================================
#  InProcessDistanceStoreAllocator
# =================================================================================================
def test_in_process_allocate_returns_a_fresh_writable_buffer():
    """In this process, a buffer is a plain float32 numpy array that the factory can fill."""
    # --- act --------------------------
    buffer = InProcessDistanceStoreAllocator().allocate((5, 5), KIND_FULL_MATRIX)

    # --- assert -----------------------
    assert buffer.shape == (5, 5)
    assert buffer.dtype == np.float32
    assert buffer.flags.writeable


def test_in_process_adopt_returns_the_array_itself():
    """In this process, adopting copies nothing: the distance store wraps the array that it was given."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act / assert -----------------
    assert InProcessDistanceStoreAllocator().adopt(vectors, KIND_LAZY, L2) is vectors


# =================================================================================================
#  SharedMemoryDistanceStoreAllocator
# =================================================================================================
def test_shared_allocate_creates_one_segment_per_call_and_records_a_spec_for_each():
    """Each allocated buffer lives in a segment of its own, described by a spec in call order."""
    # --- arrange ----------------------
    allocator = SharedMemoryDistanceStoreAllocator()

    # --- act --------------------------
    first = allocator.allocate((3, 3), KIND_FULL_MATRIX)
    second = allocator.allocate((4, 4), KIND_FULL_MATRIX)
    specs = allocator.specs
    allocator.close()

    # --- assert -----------------------
    assert [spec.shape for spec in specs] == [(3, 3), (4, 4)]
    assert specs[0].segment_name != specs[1].segment_name
    assert first.shape == (3, 3)
    assert second.shape == (4, 4)


def test_shared_adopt_copies_into_a_segment_and_reuses_it_for_the_same_array():
    """Adopting one array twice yields one segment and two specs that name it; a different array gets its own."""
    # --- arrange ----------------------
    allocator = SharedMemoryDistanceStoreAllocator()
    vectors = _vectors()
    other = _vectors() + 1.0

    # --- act --------------------------
    first = allocator.adopt(vectors, KIND_LAZY, L2)
    second = allocator.adopt(vectors, KIND_LAZY, DistanceMetric.l1_manhattan())
    third = allocator.adopt(other, KIND_LAZY, L2)
    specs = allocator.specs
    copied = np.array(first)
    allocator.close()

    # --- assert -----------------------
    assert np.shares_memory(first, second)
    assert not np.shares_memory(first, third)
    assert specs[0].segment_name == specs[1].segment_name != specs[2].segment_name
    np.testing.assert_array_equal(copied, vectors)


def test_shared_specs_carry_the_metric_of_a_lazy_store():
    """A spec for an adopted lazy array carries the metric's kind and exponent, so a worker rebuilds the same store."""
    # --- arrange ----------------------
    allocator = SharedMemoryDistanceStoreAllocator()
    metric = DistanceMetric.minkowski(3)

    # --- act --------------------------
    allocator.adopt(_vectors(), KIND_LAZY, metric)
    spec = allocator.specs[0]
    allocator.close()

    # --- assert -----------------------
    assert spec.kind == KIND_LAZY
    assert DistanceMetric(kind=spec.metric_kind, p=spec.metric_p, axis=spec.metric_axis) == metric


def test_shared_specs_attach_to_the_data_in_the_segment():
    """A spec that the allocator recorded attaches to the bytes that were written into its segment."""
    # --- arrange ----------------------
    allocator = SharedMemoryDistanceStoreAllocator()
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
    allocator.allocate((3, 3), KIND_FULL_MATRIX)[:] = matrix

    # --- act --------------------------
    with attached_distance_store(allocator.specs[0]) as store:
        seen = np.array(store.matrix)
    allocator.close()

    # --- assert -----------------------
    np.testing.assert_array_equal(seen, matrix)


def test_shared_degenerate_shape_still_gets_a_segment():
    """An empty array still gets a segment, because the operating system rejects a segment of zero bytes."""
    # --- arrange ----------------------
    allocator = SharedMemoryDistanceStoreAllocator()

    # --- act --------------------------
    buffer = allocator.allocate((0, 0), KIND_FULL_MATRIX)
    spec = allocator.specs[0]
    allocator.close()

    # --- assert -----------------------
    assert buffer.size == 0
    assert spec.shape == (0, 0)


def test_shared_close_forgets_its_segments():
    """A second close is harmless, and the specs stay as a record of what was built."""
    # --- arrange ----------------------
    allocator = SharedMemoryDistanceStoreAllocator()
    allocator.allocate((2, 2), KIND_FULL_MATRIX)

    # --- act --------------------------
    allocator.close()
    allocator.close()

    # --- assert -----------------------
    assert len(allocator.specs) == 1
