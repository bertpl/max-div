"""This module creates, attaches to, and destroys the shared-memory segments that hold one array for several processes.

`multiprocessing.shared_memory` exists on every platform that the package supports, but POSIX
leaves a segment's lifetime to the processes, which imposes two obligations: the first on every
caller of this module, the second on `attach_segment` itself.

- The process that created a segment must outlive every reader, because it is the process that
  destroys the segment.  A POSIX segment outlives its creator, and reading one through a closed
  mapping crashes the process; nothing is raised.
- An attaching process must not register with CPython's resource tracker, which is shared by the
  whole process tree; `_attach_without_registering` explains why.  The creating process does
  register, and that registration releases the segment if the creating process dies holding it.

Windows has neither concern: it keeps no tracker, its `unlink` is documented as having no effect,
and a segment goes away once the last handle to it closes.
"""

import sys
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

# `SharedMemory` accepts `track=False` from Python 3.13 on.
_TRACK_FLAG_SUPPORTED = sys.version_info >= (3, 13)


def create_segment(size_bytes: int) -> SharedMemory:
    """Create a segment of at least the given size; this process owns it and must destroy it with `destroy_segment`."""
    # the operating system rejects a segment of zero bytes, so a degenerate request still claims one byte
    return SharedMemory(create=True, size=max(size_bytes, 1))


def attach_segment(segment_name: str) -> SharedMemory:
    """Attach to an existing segment without becoming responsible for destroying it.

    The caller releases its mapping with `close()` on the returned object and never unlinks the
    segment, which belongs to the process that created it.
    """
    if _TRACK_FLAG_SUPPORTED:
        return SharedMemory(name=segment_name, track=False)
    return _attach_without_registering(segment_name)


def destroy_segment(segment: SharedMemory) -> None:
    """Close this process's mapping of a segment that this process created, and unlink the segment.

    Every mapping of the segment becomes invalid, in this process and in every process that
    attached; call this only after every reader is done.
    """
    segment.close()
    segment.unlink()


def _attach_without_registering(segment_name: str) -> SharedMemory:
    """Attach with registration suppressed, which is what `track=False` does on Python 3.13 and later.

    Registering and then unregistering would be shorter and is wrong: one tracker daemon serves the
    whole process tree, so removing the entry removes the creating process's entry too, and with it
    the cleanup that would have released the segment had the creating process died holding it.

    The suppression is process-wide for the length of one constructor call, so callers must not
    attach while another thread is creating a segment.  Windows keeps no tracker.
    """
    if sys.platform == "win32":
        return SharedMemory(name=segment_name)
    registered = resource_tracker.register
    # ty flags the assignment; replacing the module's bound method is the suppression itself
    resource_tracker.register = lambda *args, **kwargs: None  # ty: ignore[invalid-assignment]
    try:
        return SharedMemory(name=segment_name)
    finally:
        resource_tracker.register = registered
