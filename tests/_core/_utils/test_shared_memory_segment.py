from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from max_div._core._utils import attach_segment, create_segment, destroy_segment
from max_div._core._utils._shared_memory_segment import _attach_without_registering

_PAYLOAD = np.arange(6, dtype=np.float32)


def _create_with_payload() -> SharedMemory:
    """Create a segment that holds the payload array."""
    segment = create_segment(_PAYLOAD.nbytes)
    np.ndarray(_PAYLOAD.shape, dtype=np.float32, buffer=segment.buf)[:] = _PAYLOAD
    return segment


def _read_payload(segment: SharedMemory) -> np.ndarray:
    """Return a copy of the payload array as read through the given segment."""
    return np.ndarray(_PAYLOAD.shape, dtype=np.float32, buffer=segment.buf).copy()


# =================================================================================================
#  Create / attach / destroy
# =================================================================================================
def test_attached_segment_reads_what_the_creator_wrote():
    """A segment attached by name exposes the bytes that its creator wrote."""
    # --- arrange ----------------------
    owned = _create_with_payload()

    # --- act --------------------------
    attached = attach_segment(owned.name)
    values = _read_payload(attached)
    attached.close()
    destroy_segment(owned)

    # --- assert -----------------------
    np.testing.assert_array_equal(values, _PAYLOAD)


def test_closing_an_attachment_leaves_the_segment_usable():
    """Closing an attachment releases only that mapping, so the segment survives for later readers."""
    # --- arrange ----------------------
    owned = _create_with_payload()

    # --- act --------------------------
    first = attach_segment(owned.name)
    first.close()
    second = attach_segment(owned.name)
    values = _read_payload(second)
    second.close()
    destroy_segment(owned)

    # --- assert -----------------------
    np.testing.assert_array_equal(values, _PAYLOAD)


def test_destroying_a_segment_makes_its_name_unresolvable():
    """After `destroy_segment` the segment is unlinked, so attaching by its name fails."""
    # --- arrange ----------------------
    owned = _create_with_payload()
    name = owned.name

    # --- act --------------------------
    destroy_segment(owned)

    # --- assert -----------------------
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name).close()


def test_zero_bytes_still_creates_a_segment():
    """The operating system rejects a zero-byte segment, so a zero-byte request still claims a segment."""
    # --- arrange / act ----------------
    segment = create_segment(0)
    size = segment.size
    destroy_segment(segment)

    # --- assert -----------------------
    assert size >= 1


# =================================================================================================
#  Resource tracker
# =================================================================================================
def test_attaching_without_registering_reads_and_leaves_the_owner_tracked():
    """The pre-3.13 attach path maps the segment and leaves the creating process's tracker entry intact."""
    # --- arrange ----------------------
    owned = _create_with_payload()
    registered = resource_tracker.register

    # --- act --------------------------
    attached = _attach_without_registering(owned.name)
    values = _read_payload(attached)
    attached.close()
    destroy_segment(owned)  # the owner still holds its own registration, so unlinking stays clean

    # --- assert -----------------------
    np.testing.assert_array_equal(values, _PAYLOAD)
    assert resource_tracker.register is registered  # the suppression lasted one constructor call
